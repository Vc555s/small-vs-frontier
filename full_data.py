#!/usr/bin/env python3
"""Step C: the full-training-set budget (9,993 labels, seed 0) with exp1_routing.py.

  python full_data.py            train (or resume) and evaluate on the full test set
  python full_data.py --collect  copy the outputs next to the other T1 files

exp1_routing.py is run unchanged (same hyperparameters as every other T1 run). Its output
goes to results/full_data/ so that results/exp1_Qwen3-4B-Instruct-2507_s0.json (the seed-0
rows for budgets 0-2000) is never overwritten. If the adapter was saved but the evaluation
did not finish, a re-run evaluates the saved adapter instead of retraining.
Exit codes: 0 ok, 3 CUDA out of memory, 4 NaN/inf loss.
"""
import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter

DRY = os.environ.get("FINISH_DRY") == "1"   # test run: 64 labels, 1 epoch, 20 test items, separate folder
N, SEED = (64 if DRY else 9993), 0
TAG = "Qwen3-4B-Instruct-2507"
OUT = "results/finish_dryrun/full_data" if DRY else "results/full_data"
DEST = "results/finish_dryrun" if DRY else "results"


def set_paths(seed, out=None, dest=None):
    """Seed 0 keeps its original paths; other seeds (E1) pass their own output folders."""
    global SEED, OUT, DEST, ADAPTER, PREDS, RESULT
    SEED = seed
    OUT = out or OUT
    DEST = dest or DEST
    ADAPTER = os.path.join(OUT, f"adapter_{TAG}_n{N}_s{SEED}")
    PREDS = os.path.join(OUT, f"preds_{TAG}_n{N}_s{SEED}.jsonl")
    RESULT = os.path.join(OUT, f"exp1_{TAG}_s{SEED}.json")


set_paths(SEED)


def check_subset():
    from t1_common import load_data, pick_subset
    train, test, _ = load_data("mteb/banking77")
    n_full = 9993
    assert len(train) == n_full, f"train size {len(train)} != {n_full}"
    sub = pick_subset(train, n_full, SEED)
    assert Counter(sub) == Counter(train), "subset is not the whole training set exactly once"
    assert not set(sub) & set(test), "test items in the training subset"
    print(f"subset check: {len(sub)} examples = the whole training set once, no test items", flush=True)


def adapter_complete():
    return all(os.path.isfile(os.path.join(ADAPTER, f)) for f in ("adapter_config.json", "adapter_model.safetensors"))


def evaluate_saved(low_mem):
    """Evaluate the saved adapter exactly as exp1_routing.main() does after training."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    import exp1_routing as e
    from types import SimpleNamespace
    train, test, labels = e.load_data("mteb/banking77")
    tok = AutoTokenizer.from_pretrained(f"Qwen/{TAG}")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(e.load_model(f"Qwen/{TAG}", False), ADAPTER)
    model.config.use_cache = True
    args = SimpleNamespace(eval_bs=8 if low_mem else 16)
    acc, valid_rate, secs, preds = e.evaluate(model, tok, test, e.system_prompt(labels), labels, args)
    row = {"model": TAG, "budget": N, "seed": SEED, "epochs": 3, "accuracy": round(acc, 4),
           "valid_label_rate": round(valid_rate, 4), "train_seconds": None,
           "eval_seconds": round(secs), "ms_per_example": round(1000 * secs / len(test), 1),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), "test_n": len(test),
           "note": "evaluated from the saved adapter after an interrupted run; training time not recorded"}
    with open(PREDS, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    with open(RESULT, "w") as f:
        json.dump([row], f, indent=2)
    print("RESULT", json.dumps(row), flush=True)


def run(low_mem):
    import torch
    check_subset()
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(RESULT) and os.path.exists(PREDS):
        print("already complete", flush=True)
        return
    try:
        if adapter_complete():
            print(f"adapter {ADAPTER} exists; evaluating it without retraining", flush=True)
            evaluate_saved(low_mem)
        else:
            import exp1_routing as e
            argv = ["exp1_routing.py", "--budgets", str(N), "--seed", str(SEED), "--out", OUT]
            if DRY:
                argv += ["--epochs", "1", "--eval_n", "20"]
            if low_mem:
                argv += ["--bs", "2", "--accum", "4", "--eval_bs", "8"]
            sys.argv = argv
            e.main()
    except torch.OutOfMemoryError as err:
        print(f"CUDA OOM: {err}", flush=True)
        sys.exit(3)
    except RuntimeError as err:
        if "NaN" in str(err):
            print(f"NaN/inf loss: {err}", flush=True)
            sys.exit(4)
        raise


def collect(log):
    os.makedirs(DEST, exist_ok=True)
    for src, dst in [(PREDS, f"{DEST}/preds_{TAG}_n{N}_s{SEED}.jsonl"),
                     (RESULT, f"{DEST}/exp1_{TAG}_n{N}_s{SEED}.json")]:
        assert os.path.exists(src), f"missing {src}"
        assert not os.path.exists(dst) or open(src, "rb").read() == open(dst, "rb").read(), f"{dst} exists and differs"
        shutil.copyfile(src, dst)
        print(f"copied {src} -> {dst}")
    # training-loss lines (numbers only) from the stage log
    rows = []
    if log and os.path.exists(log):
        for line in open(log, errors="replace"):
            m = re.search(r"epoch (\d+)/(\d+) step (\d+)/(\d+) loss ([0-9.eE+\-naninf]+) \((\d+)s\)", line)
            if m:
                rows.append({"budget": N, "epoch": int(m.group(1)), "epochs": int(m.group(2)),
                             "step": int(m.group(3)), "total_steps": int(m.group(4)),
                             "loss": float(m.group(5)), "elapsed_s": int(m.group(6))})
    with open(f"{DEST}/train_loss_n{N}_s{SEED}.json", "w") as f:
        json.dump({"note": "Training loss as logged every 20 optimiser steps (single-batch values) by "
                           "exp1_routing.py for the full-training-set run.", "seed": SEED, "rows": rows}, f, indent=1)
    print(f"loss rows: {len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--log", default="", help="stage log to extract training loss from (with --collect)")
    ap.add_argument("--low_mem", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="working folder (adapter, raw outputs); default results/full_data")
    ap.add_argument("--dest", default="", help="where --collect copies the results; default results")
    a = ap.parse_args()
    set_paths(a.seed, a.out or None, a.dest or None)
    collect(a.log) if a.collect else run(a.low_mem)
