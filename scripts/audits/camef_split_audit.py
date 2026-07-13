"""Demonstrate CAMEF's released split behaviour using the authors' own
dataloader (unmodified) and their own sample SP500 series.

We construct date-named event folders mimicking the dataset layout (6 event
types whose dates interleave through the series span, folders created in
per-type chronological order -- the most charitable listing order), then
instantiate event_set exactly as the README's documented commands do
(no --shuffle, no --split-by-time) and once with --split-by-time.

This script is NOT self-contained: it imports the CAMEF authors' released
dataloader and reads their sample SP500 series. To run it, clone the upstream
repository and point CAMEF_REPO at the clone:

    git clone https://github.com/lakebodhi/CAMEF.git    # @ commit bd31b0c
    CAMEF_REPO=/path/to/CAMEF python scripts/audits/camef_split_audit.py

A full GPU reproduction of the split audit (on Kaggle's free T4) and its logs
are in audits/camef/ (see audits/camef/PROGRESS.md).
"""
import os, sys, shutil, importlib
# Point this at a local clone of github.com/lakebodhi/CAMEF (@ bd31b0c).
REPO = os.environ.get('CAMEF_REPO')
if not REPO:
    print("set CAMEF_REPO=/path/to/CAMEF clone "
          "(git clone https://github.com/lakebodhi/CAMEF.git @ bd31b0c)",
          file=sys.stderr)
    raise SystemExit(2)
if not os.path.isdir(REPO):
    print(f"CAMEF_REPO does not exist: {REPO} — clone "
          "https://github.com/lakebodhi/CAMEF.git and point CAMEF_REPO at it",
          file=sys.stderr)
    raise SystemExit(2)
WORK = '/tmp/camef_demo'
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
os.chdir(WORK)

# data/series with their real sample CSV
os.makedirs('data/series')
shutil.copy(f'{REPO}/dataset_sample/Time-Series/SP500.csv', 'data/series/SP500.csv')

# data/event/<type>/<date>/ tree: 6 types, dates interleaved through 2008
# (the sample series spans 2008). Type t gets one event per month, on day 3+2t.
import calendar
n_made = 0
for t in range(1, 7):
    for month in range(1, 13):
        d = f"2008{month:02d}{(3 + 2*t):02d}"
        path = f"data/event/{t}/{d}"
        os.makedirs(path, exist_ok=True)
        open(f"{path}/{d}.txt_full_summary.txt", "w").write("stub")
        n_made += 1
print(f"built {n_made} stub events across 6 interleaved types", file=sys.stderr)

sys.path.insert(0, f'{REPO}/CAMEF')
from data.dataloader import event_set

def describe(tag, ds):
    # ds.data rows: [full_summary_path, sent_reports, negs, seq, pred, event_type, (scaled...)]
    import re
    def dates_of(block):
        return sorted(int(re.search(r'(\d{8})\.txt_full_summary', r[0]).group(1)) for r in block)
    n = len(ds.data)
    b1 = [0, int(0.6*n), int(0.8*n)]
    b2 = [int(0.6*n), int(0.8*n), n]
    tr = dates_of(ds.data[b1[0]:b2[0]]); te = dates_of(ds.data[b1[1]:b2[1]]); va = dates_of(ds.data[b1[2]:b2[2]])
    print(f"\n[{tag}] n={n}  train={len(tr)} test={len(te)} vali={len(va)}")
    print(f"  train dates: {tr[0]}..{tr[-1]}   test dates: {te[0]}..{te[-1]}   vali: {va[0]}..{va[-1]}")
    leak = sum(1 for d in tr if d > te[0])
    print(f"  train events dated AFTER the earliest test event: {leak}/{len(tr)}"
          f"  ({100*leak/len(tr):.0f}% of training is the test period's future or concurrent)")
    inter = len(set(d//100 for d in tr) & set(d//100 for d in te))
    print(f"  calendar months shared by train and test: {inter}")

print("\n=== documented default (no --shuffle, no --split-by-time) ===")
ds = event_set(35, 35, series_id='SP500', shuffle=False, batch_size=10,
               event_dir='data/event', series_dir='data/series', scale=True, split_by_time=False)
describe("DEFAULT / by index", ds)

print("\n=== with --split-by-time ===")
ds2 = event_set(35, 35, series_id='SP500', shuffle=False, batch_size=10,
                event_dir='data/event', series_dir='data/series', scale=True, split_by_time=True)
describe("--split-by-time", ds2)
