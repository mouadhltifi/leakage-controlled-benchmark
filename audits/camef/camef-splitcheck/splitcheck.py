#!/usr/bin/env python3
"""
CAMEF split-leakage figure on the REAL dataset (CPU-only, no model, no GPU).

Computes, on the AUTHORS' real data via their UNMODIFIED dataloader, under the DEFAULT
(non-temporal, --shuffle off) index split: what fraction of the train-60% events have
dates AFTER the earliest test event? (The "98% on sample" figure, recomputed for real.)

We do NOT re-implement the split — we instantiate the authors' own event_set with the exact
default args and read self.data (the loader's own ordered sample list) + its 6:2:2 borders.
Faithful: dataloader.py runs byte-identical except the single num_workers token (irrelevant here,
we never build loaders). No --split-by-time, no shuffle, scale=True, seq/pred=35, SP500 — the defaults.
"""
import os, sys, subprocess, zipfile, glob, shutil, time

WORK = "/kaggle/working"; os.chdir(WORK)
def sh(c): print("$ "+" ".join(c), flush=True); return subprocess.run(c)

# minimal deps: dataloader.py needs pandas, numpy, sklearn, torch(from_numpy). All preinstalled on Kaggle.
sh([sys.executable, "-m", "pip", "install", "-q", "gdown"])
import gdown

# code
CODE = os.path.join(WORK, "CAMEF_repo")
if not os.path.isdir(CODE):
    sh(["git","clone","--depth","1","-b","publish-camef-code","https://github.com/lakebodhi/CAMEF.git",CODE])
CAMEF_DIR = os.path.join(CODE,"CAMEF")
# num_workers token edit (matches the train/eval kernels; irrelevant here since we don't call get_data)
dl=os.path.join(CAMEF_DIR,"data","dataloader.py"); s=open(dl).read()
s=s.replace("def get_data(self, data, batch_size, num_workers=15):","def get_data(self, data, batch_size, num_workers=2):")
open(dl,"w").write(s)

# dataset
ZIP=os.path.join(WORK,"dataset.zip")
if not (os.path.exists(ZIP) and os.path.getsize(ZIP)>1_000_000):
    gdown.download(id="1ejmUgVOOiHST3RjHPqsn2P3i_woGPYX_", output=ZIP, quiet=False)
EX=os.path.join(WORK,"dataset_x")
if not os.path.isdir(EX):
    os.makedirs(EX,exist_ok=True)
    with zipfile.ZipFile(ZIP) as z: z.extractall(EX)
try: os.remove(ZIP)
except OSError: pass
EVENTS=glob.glob(EX+"/**/processed_events_and_counterfactuals",recursive=True)[0]
csvs=glob.glob(EX+"/**/*.csv",recursive=True)
SERIES_DIR=os.path.dirname([p for p in csvs if os.path.basename(p).split(".")[0]=="SP500"][0])
DATA=os.path.join(CAMEF_DIR,"data"); os.makedirs(DATA,exist_ok=True)
for lk,tgt in [(os.path.join(DATA,"event"),EVENTS),(os.path.join(DATA,"series"),SERIES_DIR)]:
    if os.path.islink(lk) or os.path.exists(lk):
        try: os.remove(lk)
        except IsADirectoryError: shutil.rmtree(lk)
    os.symlink(tgt,lk)

# run from inside CAMEF_DIR so relative data/event + data/series resolve (matches the README/train kernels)
sys.path.insert(0, CAMEF_DIR); os.chdir(CAMEF_DIR)
# show the REAL os.listdir class order (this drives the positional walk)
print("os.listdir(event) class order:", [c for c in os.listdir("data/event") if os.path.isdir(os.path.join("data/event",c))], flush=True)

# instantiate the AUTHORS' event_set with the DEFAULT args (shuffle off, no split_by_time)
from data.dataloader import event_set
t0=time.time()
ds = event_set(35, 35, event_id=0, series_id="SP500", shuffle=False, batch_size=10,
               scale=True, event_dir="data/event", series_dir="data/series", split_by_time=False)
print(f"event_set built in {time.time()-t0:.0f}s", flush=True)

# self.data order == loader order; borders are the loader's own 6:2:2 (dataloader.py L163-166)
data = ds.data
n = len(data)
b_train_lo, b_train_hi = 0, int(0.6*n)
b_test_lo,  b_test_hi  = int(0.6*n), int(0.8*n)
# each item: [full_summary_path, sent_reports, neg, seq, pred, event_type, scale_seq, scale_pred]
# date = basename of full_summary up to '.txt' (e.g. .../<class>/<date>.txt/<date>.txt_full_summary.txt)
def date_of(item):
    p = item[0]
    base = os.path.basename(p)        # <date>.txt_full_summary.txt
    return int(base.split(".")[0])    # <date> as int (YYYYMMDD)
train_dates = [date_of(x) for x in data[b_train_lo:b_train_hi]]
test_dates  = [date_of(x) for x in data[b_test_lo:b_test_hi]]

earliest_test = min(test_dates)
after = sum(1 for d in train_dates if d > earliest_test)
frac = after/len(train_dates) if train_dates else float('nan')

print("="*60, flush=True)
print(f"TOTAL events (after series-range filter): n = {n}", flush=True)
print(f"train-60% events: {len(train_dates)}  | test-20% events: {len(test_dates)}", flush=True)
print(f"earliest TEST event date: {earliest_test}", flush=True)
print(f"latest TEST event date:   {max(test_dates)}", flush=True)
print(f"train date range: {min(train_dates)} .. {max(train_dates)}", flush=True)
print(f"TRAIN events with date AFTER earliest test event: {after}/{len(train_dates)}", flush=True)
print(f">>> FRACTION = {frac:.4f}  ({100*frac:.1f}%)  <<<", flush=True)
# also report a couple of robustness cuts
after_strict = sum(1 for d in train_dates if d >= earliest_test)
print(f"(>= earliest test, inclusive: {after_strict}/{len(train_dates)} = {100*after_strict/len(train_dates):.1f}%)", flush=True)
med_test = sorted(test_dates)[len(test_dates)//2]
after_med = sum(1 for d in train_dates if d > med_test)
print(f"(train events after the MEDIAN test date: {after_med}/{len(train_dates)} = {100*after_med/len(train_dates):.1f}%)", flush=True)
print("DONE", flush=True)
