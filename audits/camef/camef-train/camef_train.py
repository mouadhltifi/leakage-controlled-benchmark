#!/usr/bin/env python3
"""
CAMEF faithful reproduction on Kaggle T4 — ONE SPLIT per kernel, authors' default 5 epochs.

This kernel runs the authors' OWN train.py / test.py on their OWN dataset (downloaded from their
Google Drive) for the default SP500 Len35 config, for a SINGLE split mode selected by the env var
CAMEF_SPLIT ("index" => --split-by-time OFF; "time" => --split-by-time ON). We launch two such
kernels concurrently (one per split) so each gets its own Kaggle ~9h budget.

FAITHFULNESS: the model and training LOGIC are UNCHANGED. We keep every authors' choice, including
the 16x-per-sample contrastive RoBERTa forwards, retain_graph=True, and set_detect_anomaly(True).
Those are the runtime-cost drivers and we record them, not remove them.

DEVIATIONS FROM STOCK (env-necessary only; each documented at file:line below):
  D1. dataloader.py get_data() num_workers 15 -> 2  (Kaggle 2-vCPU box; 15 stalls / OOM-kills workers).
  D2. Accelerator = NvidiaTeslaT4 via push flag (Kaggle's default P100/sm_60 is unsupported by the
      pre-installed torch 2.10, which supports sm_75 T4). Set on `kaggle kernels push --accelerator`.
  D3. epochs = 5 (the authors' OWN train.py default; we pass it explicitly so it's on the record).
OBSERVABILITY/SAFETY ADDED (do NOT change results — they run in a side thread / wrap the subprocess):
  O1. A heartbeat thread writes /kaggle/working/heartbeat_<split>.txt every 30s (elapsed + last log tail).
  O2. The authors' train() already saves best_model.pth + lastest_model.pth EVERY epoch and prints
      per-epoch Vali/Test MSE/MAE, so even a partial run yields a usable checkpoint + metric.
  O3. A wall-clock budget guard kills the train subprocess before Kaggle's hard cap so logs persist;
      then we still run test.py on whatever checkpoint exists and print its MSE/MAE.

Headline metric the released CLI emits = MSE + MAE (+ contrastive) on the TEST split (CAMEF.test()).
That is what the paper's main tables report. -> reproduce MSE/MAE on TEST per split.
"""
import os, sys, subprocess, zipfile, glob, shutil, time, json, traceback, threading

SPLIT = os.environ.get("CAMEF_SPLIT", "index").strip().lower()
assert SPLIT in ("index", "time"), f"CAMEF_SPLIT must be 'index' or 'time', got {SPLIT!r}"
SPLIT_FLAG = [] if SPLIT == "index" else ["--split-by-time"]
EPOCHS = "5"  # D3: authors' default

WORK = "/kaggle/working"
os.chdir(WORK)
HEARTBEAT = os.path.join(WORK, f"heartbeat_{SPLIT}.txt")

LOG = []
def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    LOG.append(m)

def run(cmd, **kw):
    log("\n$ " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, **kw)
    log(f"(exit {r.returncode}, {time.time()-t0:.0f}s)")
    return r

log(f"=== CAMEF repro — SPLIT={SPLIT} (flag={SPLIT_FLAG}) epochs={EPOCHS} ===")

# ------------------------------------------------------------------ deps
# moment.yml pins torch==2.3.0 transformers==4.43.1 momentfm==0.1.4 numpy==1.25.2 accelerate==1.9.0.
# We KEEP Kaggle's pre-installed torch (reinstalling risks CUDA mismatch + is slow); on a T4 (D2) the
# pre-installed torch 2.10 supports sm_75. momentfm 0.1.4 metadata pins transformers==4.33.3, but the
# AUTHORS' moment.yml ships 4.43.1 -> we follow the authors (4.43.1) and install momentfm --no-deps so
# it does NOT drag transformers back. Faithful choice.
log("=== installing deps (authors' moment.yml; momentfm --no-deps to keep transformers 4.43.1) ===")
run([sys.executable, "-m", "pip", "install", "-q",
     "gdown", "transformers==4.43.1", "tokenizers==0.19.1", "huggingface-hub==0.24.0",
     "numpy<2", "accelerate==1.9.0", "scikit-learn==1.5.0", "sentencepiece==0.2.0"])
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "momentfm==0.1.4"])

import torch
log("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import transformers, numpy
log("transformers", transformers.__version__, "numpy", numpy.__version__)
try:
    import momentfm
    log("momentfm", getattr(momentfm, "__version__", "?"))
except Exception as e:
    log("momentfm import FAILED:", e)

# ------------------------------------------------------------------ code
log("\n=== staging CAMEF code (authors' repo, exact branch) ===")
CODE = os.path.join(WORK, "CAMEF_repo")
if not os.path.isdir(CODE):
    run(["git", "clone", "--depth", "1", "-b", "publish-camef-code",
         "https://github.com/lakebodhi/CAMEF.git", CODE])
CAMEF_DIR = os.path.join(CODE, "CAMEF")
log("CAMEF code dir:", CAMEF_DIR, "exists:", os.path.isdir(CAMEF_DIR))

# ---- D1: dataloader num_workers 15 -> 2 (CAMEF/data/dataloader.py get_data()).
dl = os.path.join(CAMEF_DIR, "data", "dataloader.py")
src = open(dl).read()
needle = "def get_data(self, data, batch_size, num_workers=15):"
assert needle in src, "num_workers signature changed upstream! re-inspect dataloader.py"
src = src.replace(needle, "def get_data(self, data, batch_size, num_workers=2):")
open(dl, "w").write(src)
log("PATCHED dataloader.py: num_workers 15 -> 2  (D1; Kaggle 2-vCPU)")

# ------------------------------------------------------------------ data
log("\n=== downloading authors' dataset.zip from Google Drive (~4 GB) ===")
import gdown
FID = "1ejmUgVOOiHST3RjHPqsn2P3i_woGPYX_"
ZIP = os.path.join(WORK, "dataset.zip")
if not (os.path.exists(ZIP) and os.path.getsize(ZIP) > 1_000_000):
    try:
        gdown.download(id=FID, output=ZIP, quiet=False)
    except Exception as e:
        log("id-form failed:", e, "-> URL form fuzzy")
        gdown.download(f"https://drive.google.com/uc?id={FID}", ZIP, quiet=False, fuzzy=True)
log(f"dataset.zip = {os.path.getsize(ZIP)/1e6:.1f} MB")

EX = os.path.join(WORK, "dataset_x")
if not os.path.isdir(EX):
    os.makedirs(EX, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        log(f"zip entries: {len(names)}")
        z.extractall(EX)

# locate events dir + series dir (lock exact paths)
csvs = glob.glob(EX + "/**/*.csv", recursive=True)
ev_candidates = glob.glob(EX + "/**/processed_events_and_counterfactuals", recursive=True)
sp = [p for p in csvs if os.path.basename(p).split(".")[0] == "SP500"]
if not ev_candidates or not sp:
    log("FATAL: could not locate events dir or SP500.csv.")
    for p in glob.glob(EX + "/**/*", recursive=True)[:120]:
        log("   ", p)
    sys.exit(2)
EVENTS = ev_candidates[0]
SERIES_DIR = os.path.dirname(sp[0])
log("EVENTS  =", EVENTS)
log("SERIES  =", SERIES_DIR)

# ---- symlink so dataloader split('/')[2] lands on the class index (ZERO code change).
DATA = os.path.join(CAMEF_DIR, "data")
os.makedirs(DATA, exist_ok=True)
for lk, tgt in [(os.path.join(DATA, "event"), EVENTS), (os.path.join(DATA, "series"), SERIES_DIR)]:
    if os.path.islink(lk) or os.path.exists(lk):
        try: os.remove(lk)
        except IsADirectoryError: shutil.rmtree(lk)
    os.symlink(tgt, lk)
    log("symlink", lk, "->", tgt)
link_event = os.path.join(DATA, "event")
probe_class = sorted([c for c in os.listdir(link_event) if os.path.isdir(os.path.join(link_event, c))])[0]
probe_sub = [s for s in os.listdir(os.path.join(link_event, probe_class))
             if os.path.isdir(os.path.join(link_event, probe_class, s))][0]
emulated = os.path.join("data/event", probe_class, probe_sub)
log(f"dataloader split('/')[2] on '{emulated}' = '{emulated.split('/')[2]}' (must equal class '{probe_class}')")
assert emulated.split('/')[2] == probe_class, "PATH MATH WRONG"

# ------------------------------------------------------------------ pre-fetch HF weights (open/ungated)
log("\n=== pre-fetching pretrained weights (all OPEN; NO LLaMA at runtime) ===")
os.environ.setdefault("HF_HOME", os.path.join(WORK, "hf"))
def hf_ok(label, fn):
    try: fn(); log("  OK:", label)
    except Exception as e: log("  FAILED:", label, "->", repr(e)[:200])
from transformers import RobertaModel, RobertaTokenizer, GPT2Model
hf_ok("roberta-base", lambda: (RobertaModel.from_pretrained("FacebookAI/roberta-base"),
                               RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")))
hf_ok("gpt2", lambda: GPT2Model.from_pretrained("openai-community/gpt2"))
def _moment():
    from momentfm import MOMENTPipeline
    m = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large",
                                       model_kwargs={'task_name': 'embedding'},
                                       cache_dir=os.path.join(WORK, "moment_model"))
    m.init()
hf_ok("AutonLab/MOMENT-1-large", _moment)

# ------------------------------------------------------------------ O1: heartbeat thread
OUT = os.path.join(WORK, f"out_{SPLIT}")
TRAIN_LOG = os.path.join(OUT, "log.txt")
_hb_stop = threading.Event()
_hb_start = time.time()
def _heartbeat():
    while not _hb_stop.is_set():
        tail = ""
        try:
            if os.path.exists(TRAIN_LOG):
                tail = " | ".join(open(TRAIN_LOG).read().splitlines()[-3:])
        except Exception:
            pass
        try:
            with open(HEARTBEAT, "w") as f:
                f.write(f"split={SPLIT} elapsed={time.time()-_hb_start:.0f}s epochs={EPOCHS}\n"
                        f"last_train_log: {tail}\n")
        except Exception:
            pass
        _hb_stop.wait(30)
threading.Thread(target=_heartbeat, daemon=True).start()

# ------------------------------------------------------------------ O3: budget-guarded train, then eval
# Kaggle GPU script kernels wall ~9h. Cap the TRAIN subprocess at 7.5h so we still have time to run
# test.py on the last checkpoint and flush logs. The authors' train() saves a checkpoint every epoch,
# so a partial run is still usable.
BUDGET_TRAIN_S = 7.5 * 3600

def train_then_eval():
    log(f"\n############### TRAIN [{SPLIT}] (epochs={EPOCHS}) ###############")
    cmd = [sys.executable, "train.py",
           "--moment-model", "AutonLab/MOMENT-1-large",
           "--event-dir", "data/event", "--series-dir", "data/series",
           "--series-id", "SP500", "--seq-len", "35", "--pred-len", "35",
           "--batch-size", "10", "--epochs", EPOCHS,
           "--output-path", OUT] + SPLIT_FLAG
    log("$ " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=CAMEF_DIR)
    walled = False
    while True:
        try:
            rc = p.wait(timeout=60)
            log(f"[{SPLIT}] train exited rc={rc} after {time.time()-t0:.0f}s")
            break
        except subprocess.TimeoutExpired:
            if time.time() - t0 > BUDGET_TRAIN_S:
                log(f"[{SPLIT}] BUDGET HIT ({BUDGET_TRAIN_S/3600:.1f}h) — terminating train to preserve logs/ckpt")
                p.terminate()
                try: p.wait(timeout=120)
                except subprocess.TimeoutExpired: p.kill()
                walled = True
                break
    # pick a checkpoint
    ckpt = os.path.join(OUT, "best_model.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(OUT, "lastest_model.pth")
    log(f"[{SPLIT}] checkpoint:", ckpt, "exists:", os.path.exists(ckpt), f"(walled={walled})")
    # surface the authors' own per-epoch metric log
    if os.path.exists(TRAIN_LOG):
        log(f"[{SPLIT}] --- authors' log.txt (last 20 lines) ---")
        for ln in open(TRAIN_LOG).read().splitlines()[-20:]:
            log("    ", ln)
    if not os.path.exists(ckpt):
        log(f"[{SPLIT}] NO checkpoint — train crashed before epoch 1; cannot eval.")
        return
    # EVAL on TEST split with the authors' test.py (the released headline metric)
    te = run([sys.executable, "test.py",
              "--checkpoint", ckpt,
              "--moment-model", "AutonLab/MOMENT-1-large",
              "--event-dir", "data/event", "--series-dir", "data/series",
              "--series-id", "SP500", "--seq-len", "35", "--pred-len", "35",
              "--batch-size", "10", "--split", "test"] + SPLIT_FLAG,
             cwd=CAMEF_DIR)
    log(f"[{SPLIT}] test.py exit {te.returncode}")

try:
    train_then_eval()
except Exception:
    log("RUN crashed:\n" + traceback.format_exc())
finally:
    _hb_stop.set()

with open(os.path.join(WORK, f"camef_repro_log_{SPLIT}.txt"), "w") as f:
    f.write("\n".join(LOG))
log(f"\nDONE [{SPLIT}]. Full log at /kaggle/working/camef_repro_log_{SPLIT}.txt")
