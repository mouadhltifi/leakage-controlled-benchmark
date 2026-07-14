"""MSGCA selection-bias diagnostic — rigorous re-run (v2).

Purpose
-------
Quantify, airtight and confound-isolated, how much MSGCA's reported headline
depends on selecting the reported checkpoint by the TEST metric instead of a
held-out validation metric. This re-run is the evidence base for the paper's
evaluation-methodology contribution.

What MSGCA's own released code does (github.com/changzong/MSGCA, main.py):
    - random 80/20 train/test split, NO validation set
    - every epoch: score the TEST set; best_mcc = max(best_mcc, test_mcc)
    - report best_mcc  ==  max test-set MCC over all epochs
So their reported number is argmax-over-epochs on the TEST set. This script
does not assume that — it is established by reading their released code. Here we
MEASURE the consequence on their own BigData22 data + their own Model.

Design (improvements over diagnostic_bias.py)
---------------------------------------------
The outer 80/20 split is FIXED at seed=42 because the doc/LLM embedding caches
in data/bd22_stock/cache/ were built for that split (regenerating them needs the
OpenAI API; the cache lets us run fully offline). Within that constraint we add:

  ARM A — Reproduction (their method): full training data, no validation set,
          report max test-MCC over epochs. Run with M model-init seeds to get a
          mean +/- std reproduction estimate (init seed 42 should match the
          original diagnostic's 0.1157 as a sanity check).

  ARM B — Within-run isolation (the clean causal number): carve 20% of the
          training entities into a validation set, train on the rest, and track
          BOTH val and test MCC every epoch. WITHIN A SINGLE RUN compare:
             best_on_test = max test-MCC over epochs        (the selection rule in the released code)
             best_on_val  = test-MCC at the best-val epoch   (validation-based criterion)
          The difference is the selection inflation with the data, the model,
          and the training run all held IDENTICAL — only the selection criterion
          differs. Run with K (val-split, init) seeds for a stable estimate.

  Budget sensitivity: every run stores full per-epoch history, so max/at-best-val
  are recomputed at epoch budgets {200, 300, 400} with no extra training. (The
  paper text says 200 epochs for BigData22; MSGCA's main.py comment says 300;
  the default is 400. Reporting all three removes the epoch-count objection.)

Also reports, for honesty, the cross-arm gap (ARM A full-data best-on-test vs
ARM B val-selected test) which bundles the selection effect with the ~20% data
carve; the WITHIN-RUN number (ARM B) is the confound-free one to headline.

This script is NOT self-contained: it runs the MSGCA authors' own released model
and data. It is included as documentation of the audit. To reproduce, clone the
upstream repository and run this script from inside the clone:

    git clone https://github.com/changzong/MSGCA.git    # @ commit 253925d
    cd MSGCA
    # (the BigData22 doc/LLM embedding caches MSGCA ships under data/bd22_stock/
    #  cache/ are required; the outer 80/20 split is fixed at seed=42 to match them)
    python /path/to/this/script/msgca_diagnostic_rerun.py            # full
    python /path/to/this/script/msgca_diagnostic_rerun.py --canary   # 5-epoch smoke

The finished re-run output that the paper's figures consume is committed at
results/analysis/msgca_diagnostic_rerun.json (+ _history.json). No OpenAI API
call is made when the caches are present; no upstream files are modified.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

# Reuse MSGCA's own data pipeline + model (no reimplementation).
from diagnostic_bias import prepare_data_original
from model import Model


# --------------------------------------------------------------------------
# argument scaffold (mirrors main.py defaults; outer seed pinned to 42)
# --------------------------------------------------------------------------
def build_args():
    p = argparse.ArgumentParser()
    # data / split (DO NOT change seed/sample_ratio: cache is keyed to them)
    p.add_argument("--data_path", type=str, default="./data/")
    p.add_argument("--dataset", type=str, default="bd22_stock")
    p.add_argument("--predict_date", type=str, default="2020-04-11")
    p.add_argument("--sample_ratio", type=float, default=0.8)
    p.add_argument("--date_move_steps", type=int, default=10)
    p.add_argument("--date_move_len", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)          # outer split seed (PINNED)
    p.add_argument("--learning_rate", type=float, default=0.0001)
    p.add_argument("--weight_decay", type=float, default=0)
    p.add_argument("--epoch_num", type=int, default=400)
    p.add_argument("--train_batch_size", type=int, default=40)
    # model dims (verbatim from main.py)
    p.add_argument("--input_graph_dim", type=int, default=64)
    p.add_argument("--output_graph_dim", type=int, default=64)
    p.add_argument("--input_llm_dim", type=int, default=1536)
    p.add_argument("--output_know_dim", type=int, default=64)
    p.add_argument("--input_ind_dim", type=int, default=1)
    p.add_argument("--output_ind_dim", type=int, default=64)
    p.add_argument("--input_doc_dim", type=int, default=128)
    p.add_argument("--output_doc_dim", type=int, default=64)
    p.add_argument("--input_att_dim", type=int, default=64)
    p.add_argument("--hidden_att_dim", type=int, default=64)
    p.add_argument("--output_att_dim", type=int, default=64)
    p.add_argument("--input_pred_dim", type=int, default=64)
    p.add_argument("--output_pred_dim", type=int, default=3)
    p.add_argument("--num_head", type=int, default=2)
    p.add_argument("--use_cuda", action="store_true", default=False)
    # re-run controls
    p.add_argument("--orig_seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="model-init seeds for ARM A (reproduction, full data)")
    p.add_argument("--val_seeds", type=int, nargs="+",
                   default=[100, 200, 300, 400, 500, 600, 700, 800],
                   help="val-split/init seeds for ARM B (within-run isolation)")
    p.add_argument("--budgets", type=int, nargs="+", default=[200, 300, 400],
                   help="epoch budgets to report max/at-best-val for")
    p.add_argument("--canary", action="store_true",
                   help="5-epoch, 1+1-seed smoke test")
    p.add_argument("--output_file", type=str,
                   default="msgca_diagnostic_rerun.json",
                   help="where to write the re-run JSON (default: CWD)")
    return p.parse_args()


def _train_eval_loop(args, model, optimizer, train_set, train_label,
                     train_node_idxs, llm_train, graph_input,
                     eval_sets, device):
    """Train args.epoch_num epochs; each epoch eval every set in eval_sets.

    eval_sets: dict name -> (set, label, node_idxs, llm). Returns per-epoch
    history: list of {epoch, train_loss, <name>_mcc, <name>_acc, ...}.
    """
    history = []
    n_train = len(train_set)
    batch_num = max(1, n_train // args.train_batch_size)
    for epoch in range(args.epoch_num):
        model.train()
        train_loss = 0.0
        for i in range(batch_num):
            s = i * args.train_batch_size
            e = min((i + 1) * args.train_batch_size, n_train)
            loss = model(train_set[s:e], graph_input, llm_train[s:e],
                         train_node_idxs[s:e], train_label[s:e], "train")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= batch_num

        rec = {"epoch": epoch, "train_loss": train_loss}
        model.eval()
        with torch.no_grad():
            for name, (eset, elab, enode, ellm) in eval_sets.items():
                acc, mcc = model(eset, graph_input, ellm, enode, elab, "test")
                rec[f"{name}_acc"] = float(acc)
                rec[f"{name}_mcc"] = float(mcc)
        history.append(rec)
    return history


def run_arm_a(args, data, init_seed, device):
    """Reproduction arm: full train data, no val, track per-epoch test."""
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    model = Model(args, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    t0 = time.time()
    hist = _train_eval_loop(
        args, model, opt,
        data["train_set"], data["train_label"], data["node_train_idxs"],
        data["llm_train"], data["graph_input"],
        {"test": (data["test_set"], data["test_label"],
                  data["node_test_idxs"], data["llm_test"])},
        device)
    return {"arm": "A_reproduction", "init_seed": init_seed,
            "n_train": len(data["train_set"]), "n_test": len(data["test_set"]),
            "elapsed_s": time.time() - t0, "history": hist}


def run_arm_b(args, data, split_seed, device):
    """Within-run isolation arm: carve 20% val from train; track val+test."""
    train_set_full = data["train_set"]
    train_label_full = data["train_label"]
    llm_train_full = data["llm_train"]
    node_train_full = data["node_train_idxs"]
    n_total = len(train_set_full)

    rng = random.Random(split_seed)
    n_val = int(n_total * 0.2)
    val_idx = set(rng.sample(range(n_total), n_val))
    tr_idx = [i for i in range(n_total) if i not in val_idx]
    val_idx = sorted(val_idx)

    train_set = [train_set_full[i] for i in tr_idx]
    val_set = [train_set_full[i] for i in val_idx]
    train_label = [train_label_full[i] for i in tr_idx]
    val_label = [train_label_full[i] for i in val_idx]
    llm_train = llm_train_full[tr_idx]
    llm_val = llm_train_full[val_idx]
    node_train = [node_train_full[i] for i in tr_idx]
    node_val = [node_train_full[i] for i in val_idx]

    # init seed tied to split seed for reproducibility, decoupled from data prep
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Model(args, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    t0 = time.time()
    hist = _train_eval_loop(
        args, model, opt,
        train_set, train_label, node_train, llm_train, data["graph_input"],
        {"val": (val_set, val_label, node_val, llm_val),
         "test": (data["test_set"], data["test_label"],
                  data["node_test_idxs"], data["llm_test"])},
        device)
    return {"arm": "B_within_run", "split_seed": split_seed,
            "n_train": len(train_set), "n_val": len(val_set),
            "n_test": len(data["test_set"]),
            "elapsed_s": time.time() - t0, "history": hist}


# --------------------------------------------------------------------------
# budget-aware analysis (recompute selection criteria at each epoch budget)
# --------------------------------------------------------------------------
def arm_a_at_budget(hist, budget):
    """Max test-MCC over epochs < budget (the released selection rule, full data)."""
    h = [r for r in hist if r["epoch"] < budget]
    best = max(h, key=lambda r: r["test_mcc"])
    return {"best_on_test": best["test_mcc"], "best_epoch": best["epoch"]}


def arm_b_at_budget(hist, budget):
    """best_on_test, best_on_val, and within-run inflation over epochs<budget."""
    h = [r for r in hist if r["epoch"] < budget]
    best_test = max(h, key=lambda r: r["test_mcc"])
    best_val = max(h, key=lambda r: r["val_mcc"])
    return {
        "best_on_test": best_test["test_mcc"],          # the released code's selection rule
        "best_on_test_epoch": best_test["epoch"],
        "best_on_val_test_mcc": best_val["test_mcc"],    # validation-based criterion
        "best_on_val_epoch": best_val["epoch"],
        "best_val_mcc": best_val["val_mcc"],
        "within_run_inflation": best_test["test_mcc"] - best_val["test_mcc"],
    }


def summarize(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0.0),
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def main():
    args = build_args()
    if args.canary:
        args.epoch_num = 5
        args.orig_seeds = [42]
        args.val_seeds = [100]
        args.budgets = [5]
        args.output_file = args.output_file.replace(".json", "_canary.json")

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} epochs={args.epoch_num} "
          f"orig_seeds={args.orig_seeds} val_seeds={args.val_seeds}")

    print("[data] loading MSGCA bd22_stock (cache-backed, no API)...")
    from data_process import load_data
    input_data, label = load_data(args, ["document", "indicator", "graph"])
    data = prepare_data_original(args, input_data, label)
    print(f"[data] {len(data['train_set'])} train / {len(data['test_set'])} test entities")

    arm_a_runs, arm_b_runs = [], []

    for s in args.orig_seeds:
        print(f"[ARM A] reproduction, init_seed={s} ...")
        r = run_arm_a(args, data, s, device)
        arm_a_runs.append(r)
        snap = arm_a_at_budget(r["history"], args.epoch_num)
        print(f"        best_on_test={snap['best_on_test']:.4f} @ep{snap['best_epoch']} "
              f"({r['elapsed_s']:.0f}s)")

    for s in args.val_seeds:
        print(f"[ARM B] within-run, split_seed={s} ...")
        r = run_arm_b(args, data, s, device)
        arm_b_runs.append(r)
        snap = arm_b_at_budget(r["history"], args.epoch_num)
        print(f"        best_on_test={snap['best_on_test']:.4f} "
              f"best_on_val_test={snap['best_on_val_test_mcc']:.4f} "
              f"inflation={snap['within_run_inflation']:+.4f} ({r['elapsed_s']:.0f}s)")

    # ---- budget-aware aggregation ----
    report = {}
    for B in args.budgets:
        a_best = [arm_a_at_budget(r["history"], B)["best_on_test"] for r in arm_a_runs]
        b = [arm_b_at_budget(r["history"], B) for r in arm_b_runs]
        report[str(B)] = {
            "arm_a_reproduction_best_on_test": summarize(a_best),
            "arm_b_best_on_test_80pct": summarize([x["best_on_test"] for x in b]),
            "arm_b_val_selected_best_on_val_test": summarize([x["best_on_val_test_mcc"] for x in b]),
            "arm_b_within_run_inflation": summarize([x["within_run_inflation"] for x in b]),
            "cross_arm_gap_repro_minus_val_selected": (
                summarize(a_best)["mean"] - summarize([x["best_on_val_test_mcc"] for x in b])["mean"]),
        }

    out = {
        "experiment": "MSGCA selection-bias diagnostic — rigorous re-run (v2)",
        "dataset": "bd22_stock",
        "outer_seed": args.seed,
        "outer_split_ratio": args.sample_ratio,
        "val_ratio": 0.2,
        "epoch_num": args.epoch_num,
        "device": str(device),
        "what_their_code_does": (
            "github.com/changzong/MSGCA main.py: random 80/20 train/test, NO val set, "
            "best_mcc = max test-MCC over epochs (argmax-over-epochs on TEST). Reported number."),
        "method": {
            "arm_a": "full train data, no val, max test-MCC over epochs (reproduces their protocol)",
            "arm_b": "carve 20% val from train; WITHIN one run compare max-test vs test-at-best-val "
                     "(isolates selection criterion; data/model/run held identical)",
            "within_run_inflation": "arm_b best_on_test - best_on_val_test_mcc (confound-free)",
            "cross_arm_gap": "arm_a best_on_test - arm_b validation-selected; bundles selection + 20% data carve",
        },
        "report_by_budget": report,
        "runs": {
            "arm_a": [{k: v for k, v in r.items() if k != "history"} for r in arm_a_runs],
            "arm_b": [{k: v for k, v in r.items() if k != "history"} for r in arm_b_runs],
        },
    }
    _out_dir = os.path.dirname(args.output_file)
    if _out_dir:  # bare filename -> write to CWD (per --output_file default)
        os.makedirs(_out_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)
    # full histories separately (large)
    hist_file = args.output_file.replace(".json", "_history.json")
    with open(hist_file, "w") as f:
        json.dump({"arm_a": [r["history"] for r in arm_a_runs],
                   "arm_b": [{"split_seed": r["split_seed"], "history": r["history"]}
                             for r in arm_b_runs]}, f)

    print("\n" + "=" * 64)
    print("SUMMARY (epoch budget -> means)")
    for B in args.budgets:
        rb = report[str(B)]
        print(f" budget {B}: repro(best-on-test)={rb['arm_a_reproduction_best_on_test']['mean']:+.4f} "
              f"| val-selected(best-on-val)={rb['arm_b_val_selected_best_on_val_test']['mean']:+.4f} "
              f"| within-run inflation={rb['arm_b_within_run_inflation']['mean']:+.4f} "
              f"(±{rb['arm_b_within_run_inflation']['std']:.4f})")
    print(f"\nsaved: {args.output_file}")
    print(f"saved: {hist_file}")


if __name__ == "__main__":
    main()
