# CAMEF Kaggle reproduction — progress & handoff

Goal: reproduce CAMEF's reported positive on Kaggle's free T4 GPU, **both splits**,
faithful to the authors' code, documenting every change needed to run it.
Feeds the paper's reproducibility audit (non-adversarial).

## Established (by two analysis passes — trust these)
- **Repo**: a local clone of lakebodhi/CAMEF @ `bd31b0c`, branch `publish-camef-code`.
- **Full dataset**: `dataset.zip` on Google Drive id `1ejmUgVOOiHST3RjHPqsn2P3i_woGPYX_`;
  `gdown` ~40s, ~40 MB. Only `dataset_sample` (~34 MB) ships in-repo.
- **Dataset layout**: event texts at
  `dataset/events/processed_events_and_counterfactuals/{1..6}/<date>.txt/` (six macro-event classes).
  Series CSVs also in the zip — confirm exact paths from the explore-kernel log.
- **Dataloader gotcha**: hardcodes `split('/')[2]` on the event path. To avoid ANY code change:
  run from INSIDE `CAMEF/` with relative `--event-dir data/event`, and symlink
  `processed_events_and_counterfactuals` → `CAMEF/data/event` (so `data/event/<class>/<date>.txt`,
  and `split('/')[2]` == the class index). Matches the README invocation. Zero code change.
- **Pretrained weights needed**: RoBERTa, GPT-2, MOMENT-1-large, LLaMA counterfactuals.
  (LLaMA may be gated — note precisely if so; do NOT substitute silently.)
- **`--split-by-time`** defaults OFF; run both OFF and ON.
- **Decision**: download+extract the dataset INSIDE the training kernel (single faithful run);
  no separate Kaggle dataset registration needed.

## Kaggle
- Account `<ANON-KAGGLE-USER>`. CLI `kaggle` v2.2.1, auth via `~/.kaggle/kaggle.json`
  (chmod 600 — never print). ~30 GPU-h/week free T4.
- **Explore kernel** (completed, non-GPU): `<ANON-KAGGLE-USER>/camef-explore`.
  Pull output: `kaggle kernels output <ANON-KAGGLE-USER>/camef-explore -p camef-explore/out`.
  Python API note: `kernels_output` returns `(files, log)` tuple; `kernels_logs(kernel) -> str` fetches the log directly.

## Next steps (mechanical)
1. Pull/read the explore log → lock exact CSV/series paths.
2. Confirm CLI + required weights from the training entrypoint/dataloader.
3. Write GPU kernel `camef-train` (enable_gpu, enable_internet): stage code, download+extract dataset,
   symlink, fetch weights, run train+eval for BOTH splits, print the paper's headline metric(s).
4. Push → monitor `kaggle kernels status` → pull outputs → report metrics per split + verdict.

## Infra note
A long-horizon **background opus agent stalled twice on the 600s stream watchdog**, both times
mid-reasoning on benign steps. Going forward, drive these mechanical Kaggle steps in the **main loop
in small increments** (or short single-shot foreground agents) — NOT a long-running background agent.

## 2026-06-15 (Camef teammate resumes)
- Prior `camef-train v1` ran to completion-of-harness but **train crashed on BOTH splits** with
  `CUDA error: no kernel image is available for execution on the device` (cudaErrorNoKernelImageForDevice).
- ROOT CAUSE (confirmed from log): kernel was assigned a **Tesla P100 (sm_60)**, and Kaggle's CURRENT
  pre-installed torch is **2.10.0+cu128** whose min supported capability has DROPPED P100. Log warning:
  "Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation."
  NOT a self-inflicted reinstall — script correctly kept Kaggle torch; Kaggle's torch just dropped P100.
- Everything else worked: deps, dataset DL+extract, symlink path-math, RoBERTa/GPT-2/MOMENT load,
  `Data Loaded: TRAIN 916 / VALI 306 / TEST 305` for BOTH index & time splits. Only the GPU kernel failed.
- FIX: force a **T4** accelerator (sm_75, supported by torch 2.10). CLI v2.2.1 supports `--accelerator`,
  which maps to `request.machine_shape` (metadata key `machine_shape`). Confirming exact token then re-push.
- Confirmed valid token from kaggle-cli docs: single T4 = `"NvidiaTeslaT4"` (default GPU = NvidiaTeslaP100).
- Re-pushed unchanged script as **camef-train v2 with `--accelerator NvidiaTeslaT4`** (script faithful;
  only the GPU type changed; keeps Kaggle's pre-installed torch 2.10 which DOES support T4 sm_75).

## 2026-06-15 — runtime assessment (lead asked: normal or hung?)
VERDICT: long is EXPECTED for this code; no evidence of a hang. No runtime LLaMA generation.
- Config (train.py): epochs default 5, our kernel passes **--epochs 10**; batch-size 10; seq/pred 35.
- Fine-tuned vs frozen (CAMEF.py train(), L497-524): TRAINED = GPT-2 transformer blocks h + ln_f,
  RoBERTa pooler, and all projection heads (bert_linear/embed/fuse/output/residual).
  FROZEN = GPT-2 wte/wpe, RoBERTa body, **all of MOMENT-1-large**. (Optimizer also lists moment/bert
  bodies at tiny LRs but train() sets their requires_grad=False, so they don't actually update.)
- NO runtime generation. dataloader only open()/reads pre-computed counterfactual .txt files
  (data_set_new.__getitem__ L38-48; sent0..9 + 5 negatives gathered as file paths L116-133).
  negative_event_based_on_type does FS walks ONCE at dataset construction, not during training.
- WHY SLOW (dominant cost): predict_batch_contrastive (CAMEF.py L189-221) calls the windowed
  RoBERTa text_embedding() **16× per sample** (1 main + 10 sent_reports + 5 negatives). At bs=10 =>
  ~160 RoBERTa forwards per train step; 92 steps/epoch => ~14.7k forwards/epoch. test() runs the SAME
  16×-path on VALI306+TEST305 AFTER EVERY EPOCH (+~9.8k forwards). Plus MOMENT-1-large fwd per batch,
  GPT-2 stack, backward with retain_graph=True AND torch.autograd.set_detect_anomaly(True) (train.py
  L?: 2-3x backward overhead). Realistic ~20-50 min/epoch on a T4 => ~3.5-8h for ONE split's 10 epochs.
- Fixed setup overhead repeats each run: deps 28s + clone 4s + **Drive download ~4 GB / ~42s** +
  extract 43,500 files + per-split dataset build ~48s. (Note: dataset.zip is ~4 GB, not the ~40 MB the
  old note guessed; 43,500 entries.)
- Kaggle GPU kernel hard cap ~9h (script self-budgets to 8h, and will SKIP the 2nd split if >45% used).

## 2026-06-15 — lead approved: kill + re-run clean, epochs=5, per-split CONCURRENT
- KILLED the epochs=10 v2 run by pushing a replacement to slug camef-train.
- New design: ONE faithful script (camef-train/camef_train.py) parametrized by env CAMEF_SPLIT
  (index|time), epochs=5 (authors' default), authors' model+training logic UNCHANGED. Added ONLY
  observability/safety: O1 heartbeat thread -> heartbeat_<split>.txt; O2 authors' per-epoch
  best/lastest ckpt (already there); O3 7.5h train budget guard then test.py on last ckpt.
  Deviations recorded in-script: D1 num_workers 15->2, D2 T4 accelerator, D3 epochs=5.
- Two per-split kernel dirs: camef-index/ (slug camef-train, code camef_index.py, env=index) and
  camef-time/ (slug camef-train-timesplit, code camef_time.py, env=time). Env-setter prepended at L2.
- PUSH index OK -> **camef-train v3 RUNNING** on T4 (this is the index/--split-by-time OFF run).
- PUSH time FAILED: "Maximum batch GPU session count of 2 reached" — the just-killed v2 still counts
  as an active GPU session transiently. Will retry the timesplit push once v2 frees its slot.

## 2026-06-15 — index RUNNING; timesplit push blocked by Kaggle API flakiness
- camef-train v3 = INDEX split (--split-by-time OFF), epochs=5, T4 — **RUNNING** (confirmed). Solid.
- Timesplit push attempts hit a mix of: "Maximum batch GPU session count of 2 reached",
  "Notebook not found", and 500 Internal Server Error on pull of the new slug. Status of both
  candidate slugs (camef-train-timesplit, camef-timesplit) = 404 (no active session). i.e. the
  "2 sessions" count is almost certainly Kaggle mis-counting during the v2->v3 teardown + general
  API instability right now — NOT a real second run. Stopped hammering to avoid spawning duplicates.
- CANONICAL timesplit slug going forward = camef-train-timesplit (camef-time/ dir holds its metadata
  + camef_time.py with env CAMEF_SPLIT=time). The stray camef-timesplit will be deleted once API is stable.
- NEXT TURN: re-check status; when a GPU slot is free + API responsive, push camef-time once; then poll
  both to COMPLETE, pull camef_repro_log_<split>.txt + authors' out_<split>/log.txt, read TEST MSE/MAE.

## 2026-06-15 — timesplit retry (lead-timed, ~20min later): STILL blocked by a GHOST GPU session
- Sanity: only camef-train (index) shows RUNNING. kernel list otherwise clean.
- Retried timesplit push to camef-train-timesplit -> "Notebook not found" (that slug got corrupted in
  the earlier thrash: 500 ISE on pull, can't push to it). Switched to fresh slugs camef-ts5 then
  camef-train-time -> both -> "Maximum batch GPU session count of 2 reached".
- Deleted ALL stray empty kernels via Python API kernels_delete(no_confirm=True): camef-timesplit,
  camef-ts5, camef-train-timesplit (and later camef-train-time). List now = camef-train + camef-explore only.
- EVEN AFTER cleanup, a fresh push STILL returns "2 sessions reached" while only ONE kernel is actually
  RUNNING and all timesplit slugs return 404 (no session). => a GHOST/lingering GPU session (most likely
  the v2 kill that Kaggle's backend hasn't released) is holding slot #2. NOT a real second run, NOT the strays.
- Per lead's instruction, STOPPED retrying to avoid spawning more placeholders. CLI delete needs interactive
  yes/no (EOFError non-interactively) -> use Python API kernels_delete(slug, no_confirm=True).
- OPTIONS for the timesplit: (1) wait for the ghost session to clear (could be a while), then push once;
  (2) SAFE FALLBACK — run timesplit SEQUENTIALLY after camef-train (index) finishes; a slot is guaranteed
  free then. The index run is the priority deliverable and is healthy. camef-time/ holds metadata
  (slug camef-train-time) + camef_time.py (env=time) ready to push the instant a slot frees.

## 2026-06-15 — lead reported index "FINISHED" but it is STILL RUNNING
- Lead got a Kaggle completion notification for camef-train. Verified directly: `kernels status` =
  KernelWorkerStatus.RUNNING across 3 consecutive polls (5s apart). `kernels output` exits 0 but
  returns ZERO files (Kaggle exposes outputs only post-completion). => the kernel has NOT finished;
  the notification was premature/mistaken.
- Did NOT push the timesplit (slot still occupied by the still-running index; avoid another
  "2 sessions" wall). Holding until index actually reaches COMPLETE.
- NEXT: when status flips to COMPLETE for real -> pull output, read camef_repro_log_index.txt +
  out_index/log.txt for TEST MSE/MAE + per-epoch wall-clock, THEN push the timesplit into the freed slot.

## 2026-06-15 — index COMPLETE (for real); timesplit LAUNCHED
- camef-train (INDEX split, --split-by-time OFF, epochs=5, T4) = KernelWorkerStatus.COMPLETE (verified).
- GHOST SESSION CLEARED the instant index actually completed -> pushed timesplit successfully:
  **camef-train-time** version 1 = RUNNING on T4 (--split-by-time ON, epochs=5, same faithful harness).
  This is the canonical timesplit slug.
- Pulling index output: `kaggle kernels output` downloads the ENTIRE /kaggle/working (4 GB dataset +
  checkpoints + git tree) so it's slow; my metric files (camef_repro_log_index.txt, out_index/log.txt)
  sit after the big dirs. Background pull to out_idx_full/ + a watcher armed for the log file.
  NOTE: `kernels files` is paginated and starts with the git tree; there is no per-file selector in CLI.
- TODO this turn: read index TEST MSE/MAE + per-epoch wall-clock once the pull lands; report timesplit ETA.

## 2026-06-15 — INDEX SPLIT RESULT (clean, epochs=5)
- Fetched metrics WITHOUT the 4 GB pull: api.kernels_output(..., file_pattern=r"(camef_repro_log_index\.txt|log\.txt)")
  returns matched small files + response.log (rendered stdout). Saved to camef-train/camef_repro_log_index.txt
  + camef-train/stdout_index_rendered.log.
- Device: Tesla T4 (sm_75) — the D2 fix worked. torch 2.10.0+cu128, transformers 4.43.1, numpy 1.26.4.
- Data paths locked: EVENTS=dataset_x/dataset/events/processed_events_and_counterfactuals ;
  SERIES=dataset_x/dataset/time-sersseries (sic). Data Loaded TRAIN 916/VALI 306/TEST 305 earlier.
- TRAIN exit rc=0 after **6799s ≈ 1.89h for 5 epochs => ~22.7 min/epoch**. best_model.pth @ epoch 5.
- **INDEX TEST (released test.py): MSE=0.0024857522, MAE=0.0425293346, Combined=1.04788, Contrastive=1.00286.**
  (Matches the epoch-5 in-train Test loss exactly; same checkpoint/split. These are on SCALED OHLC.)
- test.py ran exit 0 in 286s. NOTE warning: "Token indices sequence length longer than 512 (608>512)" —
  the windowed RoBERTa path truncates; authors' own behavior (window=500/stride=400), not our change.
- TIMESPLIT ETA: ~22.7 min/epoch x 5 + ~3 min setup + ~5 min test ≈ ~2h total for camef-train-time.

## 2026-06-15 — PAPER-REPORTED NUMBERS LOCKED (from CAMEF_Slides.pdf slide 18, RQ1 table)
- Paper RQ1 forecasting table, **SP500 (SPX), Forecasting Length = 35** (our exact config):
  **CAMEF (paper): MSE = 0.00048860, MAE = 0.0154050.**
  (SPX/35 baselines incl. ARIMA 0.0032628/0.0308016, iTransformer 0.0064082/0.0516341,
   "TEST" model 0.0073333/0.0199733 — CAMEF is the reported SOTA, lowest by design.)
- COMPARISON (index split, our faithful 5-epoch from-scratch run, released test.py):
  | metric | paper CAMEF | our repro (index, 5ep) | ratio |
  | MSE | 0.00048860 | 0.0024857522 | ~5.1x higher |
  | MAE | 0.0154050  | 0.0425293346 | ~2.8x higher |
  => trains fine, low scaled error, but does NOT reach the paper's headline under faithful 5-epoch retrain.
- KEY NUANCE for §6.2: the paper's numbers correspond to the authors' RELEASED, fully-trained checkpoint
  (Drive folder CAMEF_SP500_Len35/best_model.pth; README's test.py workflow evaluates that ckpt). We did a
  from-scratch TRAIN reproduction at the authors' default epochs=5. Two legit-but-different things: training
  reproduction vs released-weights evaluation. The paper/slides do NOT state which split (index vs time) the
  table uses; our index run is the natural match to the default. Possible follow-up: evaluate the authors'
  released checkpoint directly to separate "weights reproduce" from "training reproduces".
- Slides also confirm: 5 datasets (SPX/INDU/NDX/USGG1M/USGG5YR), horizons 35/70/140, metrics MSE/MAE,
  LLaMa3 used OFFLINE for counterfactual generation (slide 15) — consistent with our no-runtime-LLaMA finding.

## 2026-06-15 — FAITHFULNESS LEDGER (file-level, for §6.2) — index/timesplit identical harness
PUBLISHED BASELINE: lakebodhi/CAMEF, branch publish-camef-code @ bd31b0c ("Update README.md").
Local reference repo verified at bd31b0c, tracked tree byte-identical (git diff --stat empty).
Kernel staged code via: `git clone --depth 1 -b publish-camef-code https://github.com/lakebodhi/CAMEF.git`
(run log line 15) — clones that branch's HEAD (=bd31b0c). Blob hashes below are the authoritative identity.

AUTHORITATIVE git blob hashes @ bd31b0c:
  train.py            4678faf2807218e855bd6a5d28d1474eabafeb56
  test.py             63e7675e1f78935980206bff1b7922d8dc591e6a
  model/CAMEF.py      4d99dcce65a07f8b8ab4f0e92942001c5b3967a8
  data/dataloader.py  bb12fd20c9a68bb21127832d566708dd7c84bb05
sha256 of the files that ran UNMODIFIED:
  train.py        de8c2c65b7c52f8ed3c15e49b91822bf2f86124376b841392b156c1f20e29784
  test.py         932913804adcdce8d3be73ef1126d4052625bfad5097c40e1caf136f2d24ea0f
  model/CAMEF.py  56c50e8d3058bc32a95b8f32695f623b74598b2d671ad33251824f7a069717b5

FILES THE KERNEL EXECUTED:
  - OUR harness: camef-train/camef_train.py (a SEPARATE wrapper; not an authors' file). It clones the repo,
    patches ONE line of dataloader.py, stages data via symlinks, then SUBPROCESSES the authors' train.py + test.py.
  - Authors' files invoked: train.py (Popen, harness L196), test.py (Popen, harness L233). model/CAMEF.py and
    data/dataloader.py are imported by those two.

DIFF OF WHAT RAN vs PUBLISHED bd31b0c (every authors' file):
  - train.py       : IDENTICAL (subprocess only; never written). [no change]
  - test.py        : IDENTICAL (subprocess only; never written). [no change]
  - model/CAMEF.py : IDENTICAL (never written). [no change] — model defs, train()/test() loops, the 16x
                     contrastive forwards, retain_graph, contrastive loss: all byte-for-byte authors'.
  - data/dataloader.py : ONE line changed [env-necessary]:
        L314  - def get_data(self, data, batch_size, num_workers=15):
              + def get_data(self, data, batch_size, num_workers=2):
        (reproduced + diffed; the unified diff shows ONLY this hunk.)
  => [affects results]: NONE. num_workers changes only DataLoader worker-process count, not data,
     order, batching, seeds, or math. (Index split has shuffle=False; timesplit sorts chronologically.)

THE CRUX (edited-in-place vs wrapped):
  - dataloader.py was EDITED IN PLACE (harness L92-93: src.replace(...) then open(dl,"w").write(src)).
    This is the single, documented [env-necessary] edit.
  - train.py / CAMEF.py / test.py were NOT edited — invoked via subprocess.Popen.
  - Heartbeat (L180, writes /kaggle/working/heartbeat_*.txt), the budget guard (harness wraps train.py
    in Popen + a wall-clock timeout; the run finished rc=0 BEFORE any budget action — walled=False),
    and checkpointing are ALL in the wrapper / are the authors' OWN per-epoch save. They write only OUTSIDE
    the repo and do NOT touch authors' files. [observability — no effect on results]
  - The symlinks data/event, data/series (L138) are NEW files for data staging (matches README's own
    invocation `--event-dir data/event`), not edits to existing code. [env-necessary / data-staging]

epochs=5 IS the authors' published DEFAULT: train.py:29
  `parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")`
  (used at train.py:77 num_epochs=args.epochs). We passed --epochs 5 explicitly = the default.

INDEX MSE/MAE PROVENANCE: produced by the authors' OWN test.py (byte-identical), invoked on the authors'
OWN dataset (downloaded from their Drive), on the TEST split, on the best_model.pth that the authors' own
train() saved. test.py printed: "Evaluation results: ... MSE=0.0024857522062984023, MAE=0.042529334575119995".

## 2026-06-15 — SEED/REPRO CHECK (read-only) + RELEASED-CKPT EVAL launched
### Seed/reproducibility verdict (for leakage-probe feasibility)
- NO seed anywhere in the codebase. grep for random/seed/np.random/torch.manual/random_state/permutation:
  zero seeding calls. Only Python `random` (dataloader.py:4) used.
- Randomness sites: (a) dataloader.py:160 `random.shuffle(self.data)` — gated by `--shuffle` AND not split_by_time
  (line 159); (b) dataloader.py:228 `random.shuffle(event_types_with_files)` in negative_event_based_on_type.
- INDEX split membership reconstructability:
  * Default + our run = `--shuffle` OFF => self.data is NOT shuffled => train/test is a DETERMINISTIC positional
    60/20/20 split (dataloader.py:163-166) of events in os.listdir traversal order (lines 76-82; only sorted when
    split_by_time, line 143). => which samples are train vs test IS reconstructible (positional), MODULO os.listdir
    ordering (OS/filesystem-dependent but stable per run; can pin by sorting). => leakage probe FEASIBLE.
  * If `--shuffle` were ON: random.shuffle w/ no seed => NON-reproducible membership. (Not our case.)
  * negative sampling (line 228) is unseeded => counterfactual NEGATIVES differ run-to-run, but that does NOT
    change train/test membership (only contrastive inputs). Note for any exact-loss reproduction.
### Released-checkpoint eval kernel
- Built camef-eval (eval-ONLY): clones repo, stages dataset, downloads authors' RELEASED trained-models Drive
  folder (id 1nTRvr99YC_SezKXqvZ9px6sEWFW5uEvw -> CAMEF_SP500_Len35/best_model.pth + tokenizer/), runs UNMODIFIED
  test.py --split test (index, NO --split-by-time). Deviations: D1 num_workers 15->2, D2 T4. No results-affecting change.
- Pushed CONCURRENTLY with the running timesplit — **camef-eval v1 launched** (slot 2 free; ghost gone).

## 2026-06-15 — camef-eval ERROR diagnosed: DISK EXHAUSTION (env, not code/checkpoint)
- Traceback: `OSError: [Errno 28] No space left on device: 'moment_model'` during test.py MOMENT init.
- ROOT CAUSE: gdown.download_folder(WHOLE trained-models folder) pulled EVERY checkpoint. Each best_model.pth
  = **2.42 GB**, and there are 15 (5 datasets {INDU,NASDAQ,SP500,USGG1M,USGG5YR} x 3 horizons {35,70,140}).
  It downloaded ~6 (INDU x3, NASDAQ x3...) = ~15 GB before dying, ON TOP of dataset.zip 4 GB + its multi-GB
  extraction => blew Kaggle's ~20 GB working-dir quota.
- SECONDARY BUG I introduced: download died before SP500_Len35, so my "pick SP500" logic fell back to
  pths[0] = INDU_Len140 (WRONG checkpoint). Must require SP500_Len35 explicitly, no fallback.
- This is NOT a checkpoint-loadability reproducibility finding — the released ckpt never loaded on the right
  model; it died on disk. => FIX (faithful staging) + re-run, do not stop.
- FAITHFUL FIXES: (1) download ONLY the CAMEF_SP500_Len35 subfolder — exact Drive folder id from the log =
  **1Hw6LU5huj4aqKfOnWRsGcuprxmD77bUH** (contains tokenizer/ + one 2.42 GB best_model.pth);
  (2) delete dataset.zip after extraction (free ~4 GB); (3) require SP500_Len35, drop the wrong-ckpt fallback.
  No change to test.py / model. Slot free (error released it); timesplit still running.

## 2026-06-15 — camef-eval v2 re-launched (faithful disk fixes)
- Fixes applied (all env/staging, test.py+model UNCHANGED): (1) os.remove(dataset.zip) after extract (~4 GB freed);
  (2) download ONLY SP500_Len35 subfolder via gdown.download_folder(id=1Hw6LU5huj4aqKfOnWRsGcuprxmD77bUH);
  (3) require SP500_Len35 ckpt, removed the wrong-ckpt fallback; + log disk-free. py_compile OK.
- Pushed **camef-eval v2** on T4, concurrent with the still-running timesplit. Will reach RUNNING then eval.

## 2026-06-15 — RELEASED-CHECKPOINT EVAL RESULT (the headline anchor) — REPRODUCES
- camef-eval v2 COMPLETE. Fixes worked: SP500_Len35 ckpt downloaded (2419.8 MB, correct one), tokenizer present,
  disk free 13.3 GB, test.py exit 0 in 406s. The released checkpoint LOADED cleanly in the authors' OWN code
  (CAMEF.load_model_combined) — NO architecture/key mismatch. (So: not a loadability problem; loads fine.)
- Authors' RELEASED SP500_Len35 best_model.pth, UNMODIFIED test.py, index TEST split:
  test.py stdout: "Evaluation results: Combined=1.0184517252683853, MSE=0.0004310717065558256,
                   MAE=0.014936651559236791, Contrastive=1.003084002002593"
  => **RELEASED-CKPT TEST MSE=0.0004310717, MAE=0.0149366516.**
- vs PAPER-REPORTED SP500/Len35: MSE=0.00048860, MAE=0.0154050.
  => released weights REPRODUCE the headline (actually marginally BETTER: MSE 0.000431<0.000489; MAE 0.01494<0.01541;
     within ~12%/~3% — consistent w/ unseeded negative sampling + minor eval-protocol variance). HEADLINE IS REAL.
- THREE-LEG PICTURE so far:
  * Released ckpt (index)            : MSE 0.000431 / MAE 0.014937  -> reproduces paper 0.000489 / 0.015405. ✓
  * Our from-scratch 5ep (index)     : MSE 0.002486 / MAE 0.042529  -> ~5.8x worse than released ckpt.
  * From-scratch 5ep (timesplit)     : PENDING (camef-train-time running).
  => INTERPRETATION: the paper number is genuine for the AUTHORS' RELEASED WEIGHTS; the from-scratch shortfall is a
     TRAINING-BUDGET gap (their README train example uses epochs=10; released model clearly trained longer/better than
     a naive 5-epoch run). Faithful repro distinction: weights-reproduce ✓ ; default-5-epoch-train-reproduce ✗.

## 2026-06-15 — TIMESPLIT@5 RESULT + SPLIT EFFECT (decisive)
- camef-train-time COMPLETE. TRAIN rc=0 in 7579s (~25.3 min/epoch). best_model.pth @ epoch5.
- Authors' released test.py, --split-by-time ON, TEST split:
  test.py stdout: "Evaluation results: Combined=20.006335887602646, MSE=15.265559729202506,
                   MAE=3.7367620278385596, Contrastive=1.0040141305615824"
  => **TIMESPLIT@5 TEST MSE=15.2655597, MAE=3.7367620.** (in-train epoch-5 Test matched: MSE 15.27 / MAE 3.74)
- SPLIT EFFECT (random vs temporal, SAME default 5-epoch budget, same code/data):
  index@5  MSE 0.002486 / MAE 0.042529  vs  timesplit@5 MSE 15.27 / MAE 3.74
  => timesplit MSE ~6,140x HIGHER, MAE ~88x HIGHER. COLOSSAL degradation under chronological eval.
  (Note: timesplit loss INCREASES across epoch-5 steps 3.6->7.5 — diverging, not converging, on true-future.)
- => Inference (d) uses STRONG framing (not softened): the default split's low error depends on look-ahead;
  the random split is methodologically improper for time-series, and the measured effect is enormous.

## FULL THREE-LEG TABLE (all SP500/Len35, TEST split, authors' UNMODIFIED test.py + data)
  Leg                          | MSE        | MAE       | note
  Paper-reported (slides)      | 0.00048860 | 0.0154050 | RQ1 table
  Released checkpoint (index)  | 0.00043107 | 0.0149367 | REPRODUCES paper (slightly better)
  From-scratch 5ep (index)     | 0.00248575 | 0.0425293 | ~5.8x worse than released ckpt (training-budget gap)
  From-scratch 5ep (timesplit) | 15.2655597 | 3.7367620 | ~6,140x worse MSE vs index@5 (split/look-ahead effect)
  per-epoch wall-clock: index ~22.7 min/ep (6799s/5); timesplit ~25.3 min/ep (7579s/5). T4.

## 2026-06-15 — RECOMPUTE the "98%" split-leakage figure on the REAL data (CPU kernel)
- Old "98%" was on the SAMPLE w/ synthetic date grid (code-analysis). Recomputing on real data.
- Cannot reconstruct exact split membership locally: needs real os.listdir class order (explore log shows
  it is ['5','1','6','4','2','3'], NOT sorted) AND the SP500 series date-range filter (dataloader L112).
  Filename-only estimate would NOT be faithful -> run the AUTHORS' own event_set to get the EXACT figure.
- camef-splitcheck (CPU-only, no GPU, no model): instantiates the authors' unmodified event_set with DEFAULT
  args (shuffle off, split_by_time off), reads self.data (loader's own order) + its 6:2:2 borders, computes
  fraction of train-60% events with date > earliest test-event date. Also reports >=earliest and >median cuts.
  num_workers token edit present but irrelevant (we never call get_data). PUSHED, RUNNING.

## 2026-06-15 — splitcheck v1 ERROR = my own relative-path bug (NOT disk); fixed, v2 launched
- Traceback: FileNotFoundError 'data/event' at the probe print — I called os.listdir("data/event") BEFORE
  os.chdir(CAMEF_DIR), so the relative path resolved against /kaggle/working (symlink is at CAMEF_DIR/data/event).
  Disk was FINE (extraction+symlinks succeeded; dataset.zip cleanup worked). Pure harness bug, my code, not authors'.
- FIX: moved os.chdir(CAMEF_DIR) before the probe (event_set already ran post-chdir w/ relative paths). py_compile OK.
- Re-pushed **camef-splitcheck v2** (CPU-only). RUNNING.

## 2026-06-15 — REAL split-leakage figure (the "98%" recomputed) = 99.7%
- camef-splitcheck v2 COMPLETE on REAL data via authors' UNMODIFIED event_set (default: shuffle off, no split_by_time).
- n=1527 events (post SP500 series-range filter); train-60%=916, test-20%=305 — MATCHES the "Data Loaded TRAIN 916/
  TEST 305" from every training run (consistency check ✓, same split the real runs used).
- earliest TEST event = 20080116; train date range 20080103..20240411.
- **TRAIN events dated AFTER earliest test event: 913/916 = 99.7%.** (>= earliest, inclusive: also 99.7%.)
  (> median test date: 434/916 = 47.4%, as expected for a positional interleave.)
- ORDER-ROBUST: this run's os.listdir class order was ['1','3','6','5','2','4'] vs explore's ['5','1','6','4','2','3']
  (confirms os.listdir is non-deterministic across kernels) — but 99.7% is essentially order-INDEPENDENT because the
  earliest test date sits at the very start of the 2008-2024 span, so almost every event post-dates it regardless of walk order.
- VERDICT: the written "98%" should be CORRECTED to ~99.7% (913/916). Same point, slightly higher; the old 98% was the
  sample-with-synthetic-dates estimate. Figure is faithful + robust; no need to soften to qualitative.
