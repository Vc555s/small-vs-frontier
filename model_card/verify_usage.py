#!/usr/bin/env python3
"""Run the model-card usage snippet (model_card/usage.py, unmodified source) on 100 items of
results/stress/oos_mixed.jsonl and compare with the saved predictions of the same adapter
(results/stress/preds_small_S3_s0.jsonl).

  python model_card/verify_usage.py                      # adapter from the Hugging Face Hub
  python model_card/verify_usage.py --adapter <local>    # pre-upload check against a local copy
Writes results/model_card_verification.json when run against the Hub.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t1_common import load_data  # noqa: E402
from stress_common import extended_prompt, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="")
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage.py")).read()
if a.adapter:
    src = src.replace('ADAPTER = "vishal77796/qwen3-4b-banking77-router-lora"', f'ADAPTER = {a.adapter!r}')
ns = {"__name__": "usage_snippet"}
exec(compile(src, "usage.py", "exec"), ns)

_, _, labels = load_data("mteb/banking77")
prompt_identical = ns["SYSTEM"] == extended_prompt(labels)
items = read_jsonl("results/stress/oos_mixed.jsonl")[:a.n]
saved = {r["id"]: r["pred"] for r in read_jsonl("results/stress/preds_small_S3_s0.jsonl")}
preds = [ns["route"](r["text"]) for r in items]
agree = sum(p == saved[r["id"]] for p, r in zip(preds, items))
diff = [{"id": r["id"], "snippet": p, "saved": saved[r["id"]]} for p, r in zip(preds, items) if p != saved[r["id"]]]
res = {"source": "local" if a.adapter else "huggingface_hub", "n": len(items),
       "prompt_identical_to_training": prompt_identical, "agreement_with_saved": agree,
       "correct_snippet": sum(p == r["gold"] for p, r in zip(preds, items)),
       "correct_saved": sum(saved[r["id"]] == r["gold"] for r in items), "differences": diff}
print(json.dumps(res, indent=1))
if not a.adapter:
    with open("results/model_card_verification.json", "w") as f:
        json.dump(res, f, indent=1)
if not prompt_identical or agree != len(items):
    raise SystemExit("VERIFICATION FAILED")
