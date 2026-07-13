"""Statistical analysis of the publication-lag macro re-run (leakage-free).

Reads the re-run CSVs from results/macrolag/ and prints the paired deltas vs the
matched A7 baseline, plus the news-degradation and social-LSTM contrasts. Pure
pandas/scipy over the committed CSVs; no GPU or raw data needed.

Run from the artifact root:  python scripts/analysis/analyze_macrolag.py
"""
from pathlib import Path

import pandas as pd, numpy as np
from scipy import stats


def _macrolag_dir() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "results" / "macrolag").is_dir():
            return p / "results" / "macrolag"
    return Path("results/macrolag")  # CWD-relative fallback


_DIR = _macrolag_dir()


def load(arch):
    d = pd.read_csv(_DIR / f'ablation_macrolag_{arch}.csv')
    d['cfg'] = d.experiment_name.str.extract(r'macrolag_(A\d+)_')
    d['fusion'] = d.experiment_name.str.extract(r'_(concat|gated|mha)_')
    d['fold'] = d.experiment_name.str.extract(r'_f(\d+)_').astype(int)
    d['seed'] = d.experiment_name.str.extract(r'_s(\d+)$').astype(int)
    d['arch'] = arch
    return d[['cfg','fusion','fold','seed','arch','mcc']]
df = pd.concat([load('ff'), load('lstm')], ignore_index=True)

def paired_d(diff):
    diff=np.asarray(diff); m=diff.mean(); s=diff.std(ddof=1)
    t,p=stats.ttest_rel.__self__ if False else (None,None)
    t,p=stats.ttest_1samp(diff,0.0)
    return m, m/s if s>0 else 0.0, p

# A7 cell = (fold,seed,arch) mean (concat only)
a7 = df[df.cfg=='A7'].groupby(['fold','seed','arch']).mcc.mean()
print("=== RQ1: per-config paired delta vs A7  (n=30, fusion-avg within fold,seed,arch; Bonferroni k=8) ===")
OLD={'A4':-0.0014,'A5':+0.0047,'A2':-0.0068,'A9':+0.0015,'A1':-0.0051}
NM={'A4':'P+M','A5':'P+M+S','A2':'P+N+M','A9':'P+M+G','A1':'P+all'}
for c in ['A4','A5','A2','A9','A1']:
    cell=df[df.cfg==c].groupby(['fold','seed','arch']).mcc.mean()
    idx=cell.index.intersection(a7.index)
    m,d,p=paired_d((cell.loc[idx]-a7.loc[idx]).values)
    pb=min(1.0,8*p)
    print(f"  {c} {NM[c]:7s}: Δ={m:+.4f}  d={d:+.3f}  p={p:.3f}  p_bonf={pb:.3f}  (old Δ {OLD[c]:+.4f})  n={len(idx)}")

print("\n=== C4: news degradation  A2 - A4  (paired on fold,seed,fusion,arch; k=8) ===")
key=['fold','seed','fusion','arch']
a2=df[df.cfg=='A2'].set_index(key).mcc; a4=df[df.cfg=='A4'].set_index(key).mcc
idx=a2.index.intersection(a4.index)
m,d,p=paired_d((a2.loc[idx]-a4.loc[idx]).values); pb=min(1,8*p)
print(f"  overall: Δ={m:+.4f}  d={d:+.3f}  p={p:.4f}  p_bonf={pb:.4f}  n={len(idx)}   [OLD: Δ=-0.0098 d=-0.355 p_bonf=0.010]")
for a in ['ff','lstm']:
    s2=df[(df.cfg=='A2')&(df.arch==a)].set_index(['fold','seed','fusion']).mcc
    s4=df[(df.cfg=='A4')&(df.arch==a)].set_index(['fold','seed','fusion']).mcc
    i=s2.index.intersection(s4.index); m,d,p=paired_d((s2.loc[i]-s4.loc[i]).values)
    print(f"    {a}: Δ={m:+.4f}  d={d:+.3f}  p={p:.4f}  n={len(i)}")

print("\n=== C5: social-LSTM  A5(lstm) - A5(ff)  (paired on fold,seed,fusion) ===")
s5l=df[(df.cfg=='A5')&(df.arch=='lstm')].set_index(['fold','seed','fusion']).mcc
s5f=df[(df.cfg=='A5')&(df.arch=='ff')].set_index(['fold','seed','fusion']).mcc
i=s5l.index.intersection(s5f.index); m,d,p=paired_d((s5l.loc[i]-s5f.loc[i]).values)
print(f"  A5 lstm-ff: Δ={m:+.4f}  d={d:+.3f}  p={p:.4f}  n={len(i)}   [OLD: Δ=+0.0111 d=+0.439 p=0.006]")

print("\n=== §5.1: A4 raw mean (all cells) vs old highest-in-grid 0.0126 ===")
print(f"  A4 raw mean = {df[df.cfg=='A4'].mcc.mean():+.4f}   A7 raw mean = {df[df.cfg=='A7'].mcc.mean():+.4f}")
for c in ['A4','A5','A2','A9','A1']:
    print(f"    {c} raw mean = {df[df.cfg==c].mcc.mean():+.4f}")
