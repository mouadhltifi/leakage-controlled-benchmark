#!/usr/bin/env python3
"""
CAMEF RELEASED-CHECKPOINT evaluation on Kaggle T4 — anchors the paper headline.

This kernel follows the authors' README "Evaluate A Trained Model" workflow EXACTLY:
  - stage the authors' code (git clone @ publish-camef-code) + their dataset (Drive dataset.zip),
  - download the authors' RELEASED checkpoint folder CAMEF_SP500_Len35 (best_model.pth + tokenizer/)
    from their trained-models Drive folder,
  - run their UNMODIFIED test.py pointed at that checkpoint, on the index/random TEST split
    (--split-by-time OFF — the split the SP500/Len35 model corresponds to), seq/pred=35, batch=10,
  - print TEST MSE/MAE.

GOAL: does the authors' OWN released checkpoint reproduce their paper headline (≈ MSE 0.00049 / MAE 0.01541)?
This is EVAL-ONLY: no training. test.py + the checkpoint are unmodified.

DEVIATIONS FROM STOCK (env-necessary only):
  D1. dataloader.py get_data() num_workers 15 -> 2 (Kaggle 2-vCPU). Same one-token edit as the train kernels.
  D2. Accelerator = NvidiaTeslaT4 via push flag (default P100/sm_60 unsupported by pre-installed torch 2.10).
No results-affecting change. test.py and the checkpoint run byte-identical to the authors' release.
"""
import os, sys, subprocess, zipfile, glob, shutil, time, traceback

WORK = "/kaggle/working"
os.chdir(WORK)
LOG = []
def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.append(m)
def run(cmd, **kw):
    log("\n$ " + " ".join(cmd)); t0 = time.time()
    r = subprocess.run(cmd, **kw); log(f"(exit {r.returncode}, {time.time()-t0:.0f}s)"); return r

log("=== CAMEF RELEASED-CHECKPOINT eval — SP500/Len35, index TEST split ===")

# ------------------------------------------------------------------ deps (same as train kernels)
log("=== installing deps (authors' moment.yml; momentfm --no-deps to keep transformers 4.43.1) ===")
run([sys.executable, "-m", "pip", "install", "-q",
     "gdown", "transformers==4.43.1", "tokenizers==0.19.1", "huggingface-hub==0.24.0",
     "numpy<2", "accelerate==1.9.0", "scikit-learn==1.5.0", "sentencepiece==0.2.0"])
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "momentfm==0.1.4"])
import torch
log("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# ------------------------------------------------------------------ code
log("\n=== staging CAMEF code (authors' repo, exact branch) ===")
CODE = os.path.join(WORK, "CAMEF_repo")
if not os.path.isdir(CODE):
    run(["git", "clone", "--depth", "1", "-b", "publish-camef-code",
         "https://github.com/lakebodhi/CAMEF.git", CODE])
CAMEF_DIR = os.path.join(CODE, "CAMEF")
# D1: dataloader num_workers 15 -> 2
dl = os.path.join(CAMEF_DIR, "data", "dataloader.py")
src = open(dl).read()
needle = "def get_data(self, data, batch_size, num_workers=15):"
assert needle in src, "num_workers signature changed upstream!"
open(dl, "w").write(src.replace(needle, "def get_data(self, data, batch_size, num_workers=2):"))
log("PATCHED dataloader.py: num_workers 15 -> 2  (D1)")

# ------------------------------------------------------------------ dataset (Drive zip)
log("\n=== downloading authors' dataset.zip (~4 GB) ===")
import gdown
ZIP = os.path.join(WORK, "dataset.zip")
if not (os.path.exists(ZIP) and os.path.getsize(ZIP) > 1_000_000):
    gdown.download(id="1ejmUgVOOiHST3RjHPqsn2P3i_woGPYX_", output=ZIP, quiet=False)
log(f"dataset.zip = {os.path.getsize(ZIP)/1e6:.1f} MB")
EX = os.path.join(WORK, "dataset_x")
if not os.path.isdir(EX):
    os.makedirs(EX, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        log(f"zip entries: {len(z.namelist())}"); z.extractall(EX)
# free ~4 GB: extracted tree is all we need; Kaggle working-dir is disk-quota'd (~20 GB) and the
# 2.42 GB checkpoint + MOMENT cache must fit. (v1 ERRORed with No space left on device.)
try:
    os.remove(ZIP); log("freed disk: removed dataset.zip (~4 GB) after extraction")
except OSError as e:
    log("could not remove zip:", e)
csvs = glob.glob(EX + "/**/*.csv", recursive=True)
EVENTS = glob.glob(EX + "/**/processed_events_and_counterfactuals", recursive=True)[0]
SERIES_DIR = os.path.dirname([p for p in csvs if os.path.basename(p).split(".")[0] == "SP500"][0])
log("EVENTS  =", EVENTS); log("SERIES  =", SERIES_DIR)
DATA = os.path.join(CAMEF_DIR, "data"); os.makedirs(DATA, exist_ok=True)
for lk, tgt in [(os.path.join(DATA, "event"), EVENTS), (os.path.join(DATA, "series"), SERIES_DIR)]:
    if os.path.islink(lk) or os.path.exists(lk):
        try: os.remove(lk)
        except IsADirectoryError: shutil.rmtree(lk)
    os.symlink(tgt, lk); log("symlink", lk, "->", tgt)

# ------------------------------------------------------------------ RELEASED checkpoint (ONE subfolder)
# Authors' trained-models folder contains 15 checkpoints (5 datasets x 3 horizons), each best_model.pth
# is 2.42 GB. Downloading the WHOLE folder blows the Kaggle disk (v1 ERRORed: No space left on device).
# So we download ONLY the CAMEF_SP500_Len35 subfolder. Its Drive folder id (from the v1 run's folder
# listing) = 1Hw6LU5huj4aqKfOnWRsGcuprxmD77bUH ; it holds best_model.pth (2.42 GB) + tokenizer/.
log("\n=== downloading authors' RELEASED checkpoint: ONLY CAMEF_SP500_Len35 subfolder ===")
CKPT_ROOT = os.path.join(WORK, "trained_models", "CAMEF_SP500_Len35")
os.makedirs(CKPT_ROOT, exist_ok=True)
SP500_LEN35_FOLDER_ID = "1Hw6LU5huj4aqKfOnWRsGcuprxmD77bUH"
try:
    gdown.download_folder(id=SP500_LEN35_FOLDER_ID, output=CKPT_ROOT, quiet=False, use_cookies=False)
except Exception as e:
    log("download_folder(SP500_Len35) failed:", repr(e)[:200])

log("\n=== locating CAMEF_SP500_Len35/best_model.pth ===")
pths = glob.glob(CKPT_ROOT + "/**/best_model.pth", recursive=True)
log("best_model.pth found under SP500_Len35:")
for p in pths: log("   ", p, f"({os.path.getsize(p)/1e6:.1f} MB)")
# REQUIRE the SP500_Len35 checkpoint — no wrong-checkpoint fallback.
CKPT = pths[0] if pths else None
log("CHOSEN checkpoint:", CKPT)
if CKPT:
    parent = os.path.dirname(CKPT)
    log("checkpoint dir contents:", os.listdir(parent))
    tok = os.path.join(parent, "tokenizer")
    log("tokenizer/ present (needed by load_model_combined):", os.path.isdir(tok))

if not CKPT:
    log("FATAL: SP500_Len35 checkpoint not downloaded. Listing tree:")
    for p in glob.glob(CKPT_ROOT + "/**/*", recursive=True)[:200]:
        log("   ", p)
    sys.exit(2)
# free disk headroom for MOMENT cache: report free space
try:
    import shutil as _sh
    free_gb = _sh.disk_usage(WORK).free / 1e9
    log(f"disk free under {WORK}: {free_gb:.1f} GB")
except Exception:
    pass

# ------------------------------------------------------------------ EVAL (authors' UNMODIFIED test.py)
log("\n=== running authors' test.py on RELEASED checkpoint (index TEST split) ===")
te = run([sys.executable, "test.py",
          "--checkpoint", CKPT,
          "--moment-model", "AutonLab/MOMENT-1-large",
          "--event-dir", "data/event", "--series-dir", "data/series",
          "--series-id", "SP500", "--seq-len", "35", "--pred-len", "35",
          "--batch-size", "10", "--split", "test"],   # NO --split-by-time (index/random split)
         cwd=CAMEF_DIR)
log(f"test.py exit {te.returncode}")

with open(os.path.join(WORK, "camef_eval_released_log.txt"), "w") as f:
    f.write("\n".join(LOG))
log("\nDONE. Released-checkpoint eval log at /kaggle/working/camef_eval_released_log.txt")
