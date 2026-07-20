"""MSGCA diagnostic: quantify test-set selection bias.

This script runs MSGCA's own code on their bd22_stock data with one modification:
- Adds a validation split (20% of training data, split AFTER data preprocessing)
- Tracks val MCC and test MCC each epoch
- Reports both best-test-epoch MCC (their method) and val-selected test MCC (honest method)
- The gap between these two numbers quantifies the selection bias inflation.

Uses seed=42 to match cached embeddings. Varies the train/val sub-split 3 times
to get variance estimates.

Must be run from the MSGCA directory:
    cd /Users/mouadh/Thesis/work/data/external/MSGCA
    python diagnostic_bias.py
"""

import json
import random
import time
import datetime as dt
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef

from data_process import (load_data, build_input, time_aligner,
                          graph_process_static, get_emb_from_llm)
from model import Model


def prepare_data_original(args, input_data, label):
    """Reproduce original main.py data prep (lines 48-100) exactly.

    Returns train_set, test_set, train_label, test_label, graph_input,
    llm_embeddings_train, llm_embeddings_test, node_train_idxs, node_test_idxs.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Same random split as original
    idxs = list(range(len(input_data[0])))
    train_idxs = random.sample(idxs, int(len(input_data[0]) * args.sample_ratio))
    test_idxs = list(set(idxs) ^ set(train_idxs))

    # Multiple predict dates
    predict_date = dt.datetime.strptime(args.predict_date, '%Y-%m-%d')
    predict_dates = [predict_date]
    for i in range(args.date_move_steps):
        predict_date += dt.timedelta(days=args.date_move_len)
        predict_dates.append(predict_date)

    # Build train/test sets
    train_set = []
    test_set = []
    train_label = []
    test_label = []
    print("Preprocessing indicator and document data...")
    for date in predict_dates:
        train_subset = [build_input(date, input_data[i], train_idxs, 'lt')
                        for i in range(len(input_data[:-1]))]
        train_set.extend(time_aligner(train_subset, args, date, 'train'))
        test_subset = [build_input(date, input_data[i], test_idxs, 'lt')
                       for i in range(len(input_data[:-1]))]
        test_set.extend(time_aligner(test_subset, args, date, 'test'))
        train_label.extend(build_input(date, label, train_idxs, 'gt'))
        test_label.extend(build_input(date, label, test_idxs, 'gt'))

    # Trim to same time length
    time_span_train = len(train_set[0][0])
    def trim(dataset):
        out = []
        for entity in dataset:
            out.append([src[-time_span_train:] for src in entity])
        return out

    train_set = trim(train_set)
    test_set = trim(test_set)

    # Graph
    print("Processing graph data...")
    graph_input = graph_process_static(args, input_data[-1])

    n_dates = args.date_move_steps + 1
    node_train_idxs = train_idxs * n_dates
    node_test_idxs = test_idxs * n_dates

    # LLM embeddings (cached)
    print("Loading LLM embeddings...")
    time_length = len(train_set[0][0])
    llm_train = get_emb_from_llm(args, time_length, train_idxs, 'train')
    llm_test = get_emb_from_llm(args, time_length, test_idxs, 'test')

    return {
        "train_set": train_set,
        "test_set": test_set,
        "train_label": train_label,
        "test_label": test_label,
        "graph_input": graph_input,
        "llm_train": llm_train,
        "llm_test": llm_test,
        "node_train_idxs": node_train_idxs,
        "node_test_idxs": node_test_idxs,
        "train_idxs": train_idxs,
        "test_idxs": test_idxs,
    }


def run_diagnostic(args, data, val_split_seed):
    """Run one training with train/val/test split, tracking per-epoch metrics.

    The validation set is carved from the already-prepared training data
    (post-preprocessing), so we avoid any cache mismatch issues.
    """
    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"Running diagnostic with val_split_seed={val_split_seed}")
    print(f"{'='*60}")

    train_set_full = data["train_set"]
    test_set = data["test_set"]
    train_label_full = data["train_label"]
    test_label = data["test_label"]
    graph_input = data["graph_input"]
    llm_train_full = data["llm_train"]
    llm_test = data["llm_test"]
    node_train_idxs_full = data["node_train_idxs"]
    node_test_idxs = data["node_test_idxs"]

    n_total = len(train_set_full)

    # Sub-split training entities into train_proper + val
    rng = random.Random(val_split_seed)
    all_indices = list(range(n_total))
    n_val = int(n_total * 0.2)
    val_indices = set(rng.sample(all_indices, n_val))
    train_indices = [i for i in all_indices if i not in val_indices]
    val_indices = sorted(val_indices)

    train_set = [train_set_full[i] for i in train_indices]
    val_set = [train_set_full[i] for i in val_indices]
    train_label = [train_label_full[i] for i in train_indices]
    val_label = [train_label_full[i] for i in val_indices]
    llm_train = llm_train_full[train_indices]
    llm_val = llm_train_full[val_indices]
    node_train_idxs = [node_train_idxs_full[i] for i in train_indices]
    node_val_idxs = [node_train_idxs_full[i] for i in val_indices]

    print(f"Entities: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

    # Initialize model (same seed for fair comparison across splits)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Model(args, device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop with per-epoch tracking
    history = []
    best_test_mcc = -1.0
    best_test_acc = 0.0
    best_test_epoch = -1
    best_val_mcc = -1.0
    best_val_epoch = -1
    test_mcc_at_best_val = None
    test_acc_at_best_val = None

    t0 = time.time()
    for epoch in range(args.epoch_num):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_train = len(train_set)
        batch_num = max(1, n_train // args.train_batch_size)
        for i in range(batch_num):
            start = i * args.train_batch_size
            end = min((i + 1) * args.train_batch_size, n_train)

            batch_data = train_set[start:end]
            batch_llm = llm_train[start:end]
            batch_node = node_train_idxs[start:end]
            batch_label = train_label[start:end]

            loss = model(batch_data, graph_input, batch_llm,
                         batch_node, batch_label, 'train')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= batch_num

        # --- Evaluate val and test ---
        model.eval()
        with torch.no_grad():
            val_acc, val_mcc = model(
                val_set, graph_input, llm_val, node_val_idxs, val_label, 'test')
            test_acc, test_mcc = model(
                test_set, graph_input, llm_test, node_test_idxs, test_label, 'test')

        val_acc = float(val_acc)
        val_mcc = float(val_mcc)
        test_acc = float(test_acc)
        test_mcc = float(test_mcc)

        # Track best test (their method — max over all epochs)
        if test_mcc > best_test_mcc:
            best_test_mcc = test_mcc
            best_test_acc = test_acc
            best_test_epoch = epoch

        # Track best val (honest method — select epoch by val, report test at that epoch)
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_val_epoch = epoch
            test_mcc_at_best_val = test_mcc
            test_acc_at_best_val = test_acc

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mcc": val_mcc,
            "val_acc": val_acc,
            "test_mcc": test_mcc,
            "test_acc": test_acc,
        })

        if epoch % 50 == 0 or epoch == args.epoch_num - 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                  f"val_MCC={val_mcc:.4f} test_MCC={test_mcc:.4f} "
                  f"[{elapsed:.0f}s]")

    elapsed = time.time() - t0
    inflation = best_test_mcc - test_mcc_at_best_val
    honest_abs = max(abs(test_mcc_at_best_val), 0.001)

    print(f"\nCompleted {args.epoch_num} epochs in {elapsed:.0f}s")
    print(f"  Best test MCC (their method):     {best_test_mcc:.4f} at epoch {best_test_epoch}")
    print(f"  Best val MCC:                     {best_val_mcc:.4f} at epoch {best_val_epoch}")
    print(f"  Test MCC at best val (honest):    {test_mcc_at_best_val:.4f}")
    print(f"  INFLATION: {inflation:+.4f} ({inflation / honest_abs * 100:.1f}%)")

    return {
        "val_split_seed": val_split_seed,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "best_test_mcc": best_test_mcc,
        "best_test_acc": best_test_acc,
        "best_test_epoch": best_test_epoch,
        "best_val_mcc": best_val_mcc,
        "best_val_epoch": best_val_epoch,
        "test_mcc_at_best_val": test_mcc_at_best_val,
        "test_acc_at_best_val": test_acc_at_best_val,
        "inflation_mcc": inflation,
        "inflation_acc": best_test_acc - test_acc_at_best_val,
        "elapsed_s": elapsed,
        "history": history,
    }


def run_original_baseline(args, data):
    """Run the EXACT original method (no val split) for comparison."""
    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print("Running ORIGINAL method (no val split) for comparison")
    print(f"{'='*60}")

    train_set = data["train_set"]
    test_set = data["test_set"]
    train_label = data["train_label"]
    test_label = data["test_label"]
    graph_input = data["graph_input"]
    llm_train = data["llm_train"]
    llm_test = data["llm_test"]
    node_train_idxs = data["node_train_idxs"]
    node_test_idxs = data["node_test_idxs"]

    print(f"Entities: {len(train_set)} train, {len(test_set)} test (no val)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Model(args, device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_test_mcc = -1.0
    best_test_acc = 0.0
    best_test_epoch = -1
    history = []

    t0 = time.time()
    for epoch in range(args.epoch_num):
        model.train()
        train_loss = 0.0
        n_train = len(train_set)
        batch_num = max(1, n_train // args.train_batch_size)
        for i in range(batch_num):
            start = i * args.train_batch_size
            end = min((i + 1) * args.train_batch_size, n_train)
            loss = model(train_set[start:end], graph_input,
                         llm_train[start:end], node_train_idxs[start:end],
                         train_label[start:end], 'train')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= batch_num

        model.eval()
        with torch.no_grad():
            test_acc, test_mcc = model(
                test_set, graph_input, llm_test, node_test_idxs, test_label, 'test')
        test_acc = float(test_acc)
        test_mcc = float(test_mcc)

        if test_mcc > best_test_mcc:
            best_test_mcc = test_mcc
            best_test_acc = test_acc
            best_test_epoch = epoch

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_mcc": test_mcc,
            "test_acc": test_acc,
        })

        if epoch % 50 == 0 or epoch == args.epoch_num - 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                  f"test_MCC={test_mcc:.4f} [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\nOriginal method: Best test MCC = {best_test_mcc:.4f} at epoch {best_test_epoch}")

    return {
        "method": "original_no_val",
        "best_test_mcc": best_test_mcc,
        "best_test_acc": best_test_acc,
        "best_test_epoch": best_test_epoch,
        "elapsed_s": elapsed,
        "history": history,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument("--dataset", type=str, default='bd22_stock')
    parser.add_argument("--predict_date", type=str, default='2020-04-11')
    parser.add_argument("--sample_ratio", type=float, default=0.8)
    parser.add_argument("--date_move_steps", type=int, default=10)
    parser.add_argument("--date_move_len", type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--epoch_num", type=int, default=400)
    parser.add_argument("--train_batch_size", type=int, default=40)
    parser.add_argument("--input_graph_dim", type=int, default=64)
    parser.add_argument("--output_graph_dim", type=int, default=64)
    parser.add_argument("--input_llm_dim", type=int, default=1536)
    parser.add_argument("--output_know_dim", type=int, default=64)
    parser.add_argument("--input_ind_dim", type=int, default=1)
    parser.add_argument("--output_ind_dim", type=int, default=64)
    parser.add_argument("--input_doc_dim", type=int, default=128)
    parser.add_argument("--output_doc_dim", type=int, default=64)
    parser.add_argument("--input_att_dim", type=int, default=64)
    parser.add_argument("--hidden_att_dim", type=int, default=64)
    parser.add_argument("--output_att_dim", type=int, default=64)
    parser.add_argument("--input_pred_dim", type=int, default=64)
    parser.add_argument("--output_pred_dim", type=int, default=3)
    parser.add_argument("--num_head", type=int, default=2)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument("--output_file", type=str,
                        default="/Users/mouadh/Thesis/work/experiments/phase4r/analysis/msgca_diagnostic.json")
    args = parser.parse_args()

    print("Loading MSGCA bd22_stock data...")
    sources = ['document', 'indicator', 'graph']
    input_data, label = load_data(args, sources)
    print(f"Loaded: {len(input_data)} sources, {len(input_data[0])} stocks")

    # Prepare data using original pipeline (preserves cache compatibility)
    data = prepare_data_original(args, input_data, label)
    print(f"\nPrepared: {len(data['train_set'])} train entities, "
          f"{len(data['test_set'])} test entities")

    # Run original method (no val split) for reproduction check
    original_result = run_original_baseline(args, data)

    # Run 3 different train/val sub-splits
    val_results = []
    for split_seed in [100, 200, 300]:
        result = run_diagnostic(args, data, val_split_seed=split_seed)
        val_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY: Test-Set Selection Bias Quantification")
    print("=" * 70)
    print(f"\nOriginal method (no val): Best test MCC = {original_result['best_test_mcc']:.4f} "
          f"at epoch {original_result['best_test_epoch']}")
    print(f"\nWith validation split (20% of train):")
    print(f"{'Split':>8s} | {'Best Test MCC':>14s} | {'Val-Sel Test MCC':>17s} | {'Inflation':>10s}")
    print("-" * 60)
    for r in val_results:
        print(f"{r['val_split_seed']:>8d} | {r['best_test_mcc']:>14.4f} | "
              f"{r['test_mcc_at_best_val']:>17.4f} | {r['inflation_mcc']:>+10.4f}")

    mean_best = np.mean([r["best_test_mcc"] for r in val_results])
    mean_honest = np.mean([r["test_mcc_at_best_val"] for r in val_results])
    mean_inflation = np.mean([r["inflation_mcc"] for r in val_results])

    print("-" * 60)
    print(f"{'Mean':>8s} | {mean_best:>14.4f} | {mean_honest:>17.4f} | {mean_inflation:>+10.4f}")

    honest_abs = max(abs(mean_honest), 0.001)
    print(f"\nKey findings:")
    print(f"  Original reported MCC:        {original_result['best_test_mcc']:.4f}")
    print(f"  With proper model selection:   {mean_honest:.4f}")
    print(f"  Inflation from test-peeking:   {mean_inflation:+.4f} "
          f"({mean_inflation / honest_abs * 100:.1f}%)")
    print(f"  Best test epoch (original):    {original_result['best_test_epoch']}")
    print(f"  Best val epoch (mean):         {np.mean([r['best_val_epoch'] for r in val_results]):.0f}")

    # Save results
    output = {
        "experiment": "MSGCA test-set selection bias diagnostic",
        "dataset": "bd22_stock",
        "outer_seed": 42,
        "outer_split_ratio": 0.8,
        "val_ratio": 0.2,
        "n_epochs": args.epoch_num,
        "device": "cpu",
        "methodology": {
            "their_method": "max(test_MCC) across all 400 epochs, no validation set",
            "honest_method": "test_MCC at epoch with best val_MCC, proper model selection",
            "inflation": "best_test_MCC minus val_selected_test_MCC",
            "approach": "Use original data pipeline (cached embeddings with seed=42), "
                        "then sub-split train entities into 80% train_proper + 20% val. "
                        "Three different sub-splits provide variance.",
        },
        "original_baseline": {
            k: v for k, v in original_result.items() if k != "history"
        },
        "val_split_results": [
            {k: v for k, v in r.items() if k != "history"} for r in val_results
        ],
        "summary": {
            "original_best_test_mcc": original_result["best_test_mcc"],
            "mean_best_test_mcc_with_val_split": mean_best,
            "mean_val_selected_test_mcc": mean_honest,
            "mean_inflation_mcc": mean_inflation,
            "mean_inflation_pct": mean_inflation / honest_abs * 100,
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    # Save epoch-level history
    history_file = args.output_file.replace('.json', '_history.json')
    history_data = {
        "original": original_result["history"],
        "val_splits": [
            {"val_split_seed": r["val_split_seed"], "history": r["history"]}
            for r in val_results
        ],
    }
    with open(history_file, 'w') as f:
        json.dump(history_data, f)
    print(f"Epoch history saved to {history_file}")


if __name__ == "__main__":
    main()
