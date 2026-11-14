#!/usr/bin/env python3
"""Count the manual judgements in results/next/joint_errors_for_review.csv, split by which
system was wrong (Claude zero-shot only, Qwen 9,993 seed 0 only, or both). Refuses to run while
any judgement cell is empty or holds a value other than label_wrong, ambiguous or models_wrong.

  python summarise_review.py [path]
"""
import csv
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/next/joint_errors_for_review.csv"
VALUES = ("label_wrong", "ambiguous", "models_wrong")
GROUPS = ("Claude only", "Qwen only", "Both")

rows = list(csv.DictReader(open(PATH, newline="")))
empty = [r["test_idx"] for r in rows if not r["judgement"].strip()]
if empty:
    sys.exit(f"Refusing to summarise: {len(empty)} of {len(rows)} judgement cells are empty "
             f"(test_idx {', '.join(empty)}).")
bad = [(r["test_idx"], r["judgement"]) for r in rows if r["judgement"].strip() not in VALUES]
if bad:
    sys.exit(f"Refusing to summarise: unknown judgement values {bad}; allowed: {', '.join(VALUES)}.")

counts = {g: dict.fromkeys(VALUES, 0) for g in GROUPS}
for r in rows:
    q_wrong = r["qwen_9993_s0_pred"] != r["gold"]
    c_wrong = r["claude_zeroshot_pred"] != r["gold"]
    if not (q_wrong or c_wrong):
        sys.exit(f"Refusing to summarise: test_idx {r['test_idx']} has no wrong prediction.")
    group = "Both" if q_wrong and c_wrong else ("Qwen only" if q_wrong else "Claude only")
    counts[group][r["judgement"].strip()] += 1

print(f"{PATH}: {len(rows)} messages\n")
print("| Systems wrong | " + " | ".join(VALUES) + " | Total |")
print("|---|" + "---|" * (len(VALUES) + 1))
for g in GROUPS:
    c = counts[g]
    print(f"| {g} | " + " | ".join(str(c[v]) for v in VALUES) + f" | {sum(c.values())} |")
tot = {v: sum(counts[g][v] for g in GROUPS) for v in VALUES}
print("| All | " + " | ".join(str(tot[v]) for v in VALUES) + f" | {len(rows)} |")
