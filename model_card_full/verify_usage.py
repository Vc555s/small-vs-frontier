#!/usr/bin/env python3
"""Run the model-card usage snippet (model_card_full/usage.py, unmodified source) on the first
100 messages of results/next/clean500.jsonl (the 500-message sample in batch order) and compare
with the saved full-test predictions of the same adapter
(results/preds_Qwen3-4B-Instruct-2507_n9993_s0.jsonl, one row per test index).

  python model_card_full/verify_usage.py                      # adapter from the Hugging Face Hub
  python model_card_full/verify_usage.py --adapter <local>    # pre-upload check against a local copy
Writes results/model_card_full_verification.json when run against the Hub.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t1_common import load_data, system_prompt  # noqa: E402
from stress_common import read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="")
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage.py")).read()
if a.adapter:
    src = src.replace('ADAPTER = "vishal77796/qwen3-4b-banking77-full-lora"', f'ADAPTER = {a.adapter!r}')
ns = {"__name__": "usage_snippet"}
exec(compile(src, "usage.py", "exec"), ns)

_, test, labels = load_data("mteb/banking77")
prompt_identical = ns["SYSTEM"] == system_prompt(labels)
items = read_jsonl("results/next/clean500.jsonl")[:a.n]
saved_rows = read_jsonl("results/preds_Qwen3-4B-Instruct-2507_n9993_s0.jsonl")
assert len(saved_rows) == len(test) and all(r["text"] == t for r, (t, _) in zip(saved_rows, test))
saved = {r["src_idx"]: saved_rows[r["src_idx"]]["pred"] for r in items}
preds = [ns["route"](r["text"]) for r in items]
agree = sum(p == saved[r["src_idx"]] for p, r in zip(preds, items))
diff = [{"test_idx": r["src_idx"], "snippet": p, "saved": saved[r["src_idx"]]}
        for p, r in zip(preds, items) if p != saved[r["src_idx"]]]
res = {"source": "local" if a.adapter else "huggingface_hub", "n": len(items),
       "prompt_identical_to_training": prompt_identical, "agreement_with_saved": agree,
       "correct_snippet": sum(p == r["gold"] for p, r in zip(preds, items)),
       "correct_saved": sum(saved[r["src_idx"]] == r["gold"] for r in items), "differences": diff}
print(json.dumps(res, indent=1))
if not a.adapter:
    with open("results/model_card_full_verification.json", "w") as f:
        json.dump(res, f, indent=1)
if not prompt_identical or agree != len(items):
    raise SystemExit("VERIFICATION FAILED")
