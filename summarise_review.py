#!/usr/bin/env python3
"""Count the manual judgements in results/next/joint_errors_for_review.csv, split by which
system was wrong (Claude zero-shot only, Qwen 9,993 seed 0 only, or both). Refuses to run while
any judgement cell is empty or holds a value other than label_wrong, ambiguous or models_wrong.

  python summarise_review.py [path]

The report scripts import review_caveat() for their caveat sentence.
"""
import csv
import sys

PATH = "results/next/joint_errors_for_review.csv"
VALUES = ("label_wrong", "ambiguous", "models_wrong")
GROUPS = ("Claude only", "Qwen only", "Both")


class IncompleteReview(Exception):
    pass


def review_counts(path=PATH):
    """{group: {judgement: count}} and the number of rows; raises IncompleteReview if any cell is missing."""
    rows = list(csv.DictReader(open(path, newline="")))
    empty = [r["test_idx"] for r in rows if not r["judgement"].strip()]
    if empty:
        raise IncompleteReview(f"{len(empty)} of {len(rows)} judgement cells are empty "
                               f"(test_idx {', '.join(empty)}).")
    bad = [(r["test_idx"], r["judgement"]) for r in rows if r["judgement"].strip() not in VALUES]
    if bad:
        raise IncompleteReview(f"unknown judgement values {bad}; allowed: {', '.join(VALUES)}.")
    counts = {g: dict.fromkeys(VALUES, 0) for g in GROUPS}
    for r in rows:
        q_wrong = r["qwen_9993_s0_pred"] != r["gold"]
        c_wrong = r["claude_zeroshot_pred"] != r["gold"]
        if not (q_wrong or c_wrong):
            raise IncompleteReview(f"test_idx {r['test_idx']} has no wrong prediction.")
        group = "Both" if q_wrong and c_wrong else ("Qwen only" if q_wrong else "Claude only")
        counts[group][r["judgement"].strip()] += 1
    return counts, len(rows)


def review_caveat(path=PATH):
    """Caveat sentence for the reports, or None while the review is incomplete."""
    try:
        counts, _ = review_counts(path)
    except (IncompleteReview, FileNotFoundError):
        return None
    both = counts["Both"]
    return (f"Of the {sum(both.values())} messages both systems got wrong, a manual review judged "
            f"{both['label_wrong']} to have a wrong gold label, {both['ambiguous']} to be ambiguous and "
            f"{both['models_wrong']} to be a genuine error by both systems.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else PATH
    try:
        counts, n = review_counts(path)
    except IncompleteReview as e:
        sys.exit(f"Refusing to summarise: {e}")
    print(f"{path}: {n} messages\n")
    print("| Systems wrong | " + " | ".join(VALUES) + " | Total |")
    print("|---|" + "---|" * (len(VALUES) + 1))
    for g in GROUPS:
        c = counts[g]
        print(f"| {g} | " + " | ".join(str(c[v]) for v in VALUES) + f" | {sum(c.values())} |")
    tot = {v: sum(counts[g][v] for g in GROUPS) for v in VALUES}
    print("| All | " + " | ".join(str(tot[v]) for v in VALUES) + f" | {n} |")


if __name__ == "__main__":
    main()
