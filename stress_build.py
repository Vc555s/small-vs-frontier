#!/usr/bin/env python3
"""Stage `build`: create and freeze the T1 robustness test sets.

Outputs (in --dir):
  typo500.jsonl        Test A  (500 base messages with typos, p = 0.15)
  merged250.jsonl      Test B  (pairs of base messages with different intents)
  oos_mixed.jsonl      Test C  (500 base messages + 200 CLINC oos test messages)
  s3_train_oos.jsonl   200 CLINC oos train messages for S3 training
  s2_val_s{0,1,2}.jsonl  S2 threshold-selection sets (500 in-scope train + 100 oos)
  build_log.json       counts, every removed out-of-scope item, typo examples, decisions
All rules and seeds follow the robustness test plan (see STRESS_DECISIONS.md and
results/t1_robustness.md). Nothing here depends on any model output.
"""
import argparse
import json
import os
import random
import re
import subprocess

from exp1_routing import load_data, pick_subset
from stress_common import NONE_LABEL, SEEDS, write_json_atomic, write_jsonl_atomic

TYPO_P = 0.15
CLINC_NAME, CLINC_CONFIG, OOS_NAME = "clinc/clinc_oos", "plus", "oos"
BANKING_KEYWORDS = (
    "card, cards, account, accounts, bank, banking, transfer, transfers, payment, payments, pay, "
    "paid, money, cash, atm, pin, refund, top up, top-up, topup, currency, currencies, exchange, "
    "balance, deposit, withdraw, withdrawal, charge, charged, fee, fees, identity, verify, passcode, "
    "statement, transaction, transactions, direct debit, cheque, check, visa, mastercard, apple pay, "
    "google pay, wallet, credit, debit, loan, bill, bills, purchase, spending")
KW = [k.strip() for k in BANKING_KEYWORDS.split(",")]
KW_RE = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(k) for k in KW) + r")(?![a-z0-9])")

# QWERTY neighbours (same row and adjacent rows), letters only.
QWERTY = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj",
    "i": "uojk", "o": "ipkl", "p": "ol", "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop", "z": "asx", "x": "zsdc",
    "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}
OPS = ["swap", "delete", "duplicate", "neighbour"]


def banking_hit(text):
    m = KW_RE.search(text.lower())
    return m.group(0) if m else None


def typo_word(w, rng, log):
    letters = sum(c.isalpha() for c in w)
    if letters < 4 or any(c.isdigit() for c in w):
        return w
    if rng.random() >= TYPO_P:
        return w
    op = rng.choice(OPS)
    inner = [i for i in range(1, len(w) - 1) if w[i].isalpha()]
    if op == "swap":
        cand = [i for i in inner if i + 1 in inner]
        if not cand:
            log.append(("no_swap_position", w))
            return w
        i = rng.choice(cand)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    i = rng.choice(inner)
    if op == "delete":
        return w[:i] + w[i + 1:]
    if op == "duplicate":
        return w[:i] + w[i] + w[i:]
    nb = QWERTY.get(w[i])
    if not nb:
        log.append(("no_neighbour", w))
        return w
    return w[:i] + rng.choice(nb) + w[i + 1:]


def make_typo(text, idx, log):
    rng = random.Random(1000 + idx)
    t = text.lower()
    t = re.sub(r"[^\w\s']|_", "", t)          # drop punctuation except apostrophes
    words = t.split()
    return " ".join(typo_word(w, rng, log) for w in words)


def lower_first(s):
    return s[:1].lower() + s[1:] if s else s


def build_merged(base, test):
    """Shuffle with Random(789), pair off, reject-and-redraw equal-label pairs."""
    order = base[:]
    random.Random(789).shuffle(order)
    pairs, leftovers = [], []
    for k in range(0, len(order) - 1, 2):
        a, b = order[k], order[k + 1]
        if test[a][1] == test[b][1]:
            leftovers += [a, b]
        else:
            pairs.append((a, b))
    if len(order) % 2:
        leftovers.append(order[-1])
    # Re-pair leftovers among themselves: greedily take the first remaining partner
    # with a different label, in shuffled order. Unpairable ones are dropped.
    dropped = []
    rest = leftovers[:]
    while rest:
        a = rest.pop(0)
        j = next((j for j, b in enumerate(rest) if test[b][1] != test[a][1]), None)
        if j is None:
            dropped.append(a)
        else:
            pairs.append((a, rest.pop(j)))
    return pairs, leftovers, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/stress")
    ap.add_argument("--commit", action="store_true", help="git-commit the frozen sets")
    a = ap.parse_args()
    d = a.dir
    os.makedirs(d, exist_ok=True)
    log = {"decisions": [], "counts": {}}

    train_all, test, labels = load_data("mteb/banking77")
    base = json.load(open("results/sample500.json"))["indices"]
    order = json.load(open("results/sample500_batch_order.json"))["order"]
    assert sorted(order) == sorted(base) and len(base) == 500

    # ------------------------------------------------------------- Test A
    tlog = []
    typo_rows = [{"id": f"A{i}", "text": make_typo(test[i][0], i, tlog), "gold": test[i][1],
                  "src_idx": i, "orig_text": test[i][0]} for i in order]
    changed = sum(r["text"] != re.sub(r"\s+", " ", re.sub(r"[^\w\s']|_", "", r["orig_text"].lower())).strip()
                  for r in typo_rows)
    log["counts"]["typo500"] = len(typo_rows)
    log["counts"]["typo_messages_with_at_least_one_edit"] = changed
    log["typo_edit_fallbacks"] = tlog
    ex_ids = sorted(random.Random(15).sample(range(len(typo_rows)), 15))
    log["typo_examples"] = [{"src_idx": typo_rows[k]["src_idx"], "before": typo_rows[k]["orig_text"],
                             "after": typo_rows[k]["text"]} for k in ex_ids]
    log["decisions"].append(
        "Test A rows are stored in the T1 Claude batch order (results/sample500_batch_order.json, "
        "Random(456)), so Claude's typo batches contain the same messages as its clean T1 batches.")

    # ------------------------------------------------------------- Test B
    pairs, leftovers, dropped = build_merged(base, test)
    merged_rows = []
    for k, (ia, ib) in enumerate(pairs):
        merged_rows.append({"id": f"B{k}", "text": f"{test[ia][0]} Also, {lower_first(test[ib][0])}",
                            "gold": [test[ia][1], test[ib][1]], "gold_a": test[ia][1],
                            "gold_b": test[ib][1], "src_idx_a": ia, "src_idx_b": ib})
    log["counts"]["merged_pairs"] = len(merged_rows)
    log["counts"]["merged_equal_label_pairs_redrawn"] = len(leftovers) // 2
    log["counts"]["merged_dropped"] = len(dropped)
    log["merged_dropped_idx"] = dropped
    log["decisions"].append(
        "Test B re-pairing: items from equal-label pairs (and any odd item) are re-paired greedily "
        "in shuffled order with the first remaining item of a different label; unpairable items are dropped.")

    # ------------------------------------------------------------- Test C data
    c_ok, c_reason = True, ""
    try:
        from datasets import load_dataset
        clinc = load_dataset(CLINC_NAME, CLINC_CONFIG)
        names = clinc["train"].features["intent"].names
        oos_id = names.index(OOS_NAME)
    except Exception as e:  # noqa: BLE001
        c_ok, c_reason = False, f"could not load {CLINC_NAME}/{CLINC_CONFIG} with an '{OOS_NAME}' intent: {e!r}"

    if c_ok:
        log["decisions"].append(
            f"CLINC source: {CLINC_NAME}, config '{CLINC_CONFIG}' (the config with the most oos "
            f"train examples: 250; 'small' and 'imbalanced' have 100). oos intent id {oos_id}.")
        removed = []

        def draw(split, seed, n, exclude=(), tag=""):
            pool = [i for i, x in enumerate(clinc[split]) if x["intent"] == oos_id and i not in exclude]
            random.Random(seed).shuffle(pool)
            kept = []
            for i in pool:
                if len(kept) == n:
                    break
                t = clinc[split][i]["text"]
                hit = banking_hit(t)
                if hit:
                    removed.append({"group": tag, "split": split, "clinc_idx": i, "text": t, "keyword": hit})
                else:
                    kept.append(i)
            return kept, pool

        test_oos, _ = draw("test", 321, 200, tag="test")
        s3_oos, s3_pool = draw("train", 654, 200, tag="s3_train")
        # S2 validation: the 'plus' train split has too few oos items left after S3
        # (235 survive the filter; 200 go to S3). Draw the 100 from the remaining train
        # oos items plus the validation-split oos items, never from test.
        val_pool = ([("train", i) for i in s3_pool if i not in s3_oos]
                    + [("validation", i) for i, x in enumerate(clinc["validation"]) if x["intent"] == oos_id])
        random.Random(655).shuffle(val_pool)
        s2_oos = []
        for sp, i in val_pool:
            if len(s2_oos) == 100:
                break
            t = clinc[sp][i]["text"]
            hit = banking_hit(t)
            if hit:
                removed.append({"group": "s2_val", "split": sp, "clinc_idx": i, "text": t, "keyword": hit})
            else:
                s2_oos.append((sp, i))
        log["decisions"].append(
            "S2 validation oos items: 100 drawn with Random(655) from the 'plus' train oos items not "
            "used by S3 plus the 'plus' validation-split oos items, because the train split alone "
            "cannot supply 200 (S3) + 100 (S2) after filtering. The test split is never used for either.")
        # Removed items are only logged for the items actually inspected while drawing.
        log["removed_out_of_scope"] = removed
        log["counts"].update({"oos_test": len(test_oos), "oos_s3_train": len(s3_oos),
                              "oos_s2_val": len(s2_oos), "oos_removed_by_filter": len(removed)})
        texts = {}
        for grp, items in (("test", [("test", i) for i in test_oos]), ("s3", [("train", i) for i in s3_oos]),
                           ("s2", s2_oos)):
            for sp, i in items:
                key = (sp, i)
                assert key not in texts, f"overlap between oos groups: {key}"
                texts[key] = grp
        if len(test_oos) < 200 or len(s3_oos) < 200 or len(s2_oos) < 100:
            log["decisions"].append(
                f"Fewer oos items than planned survived the filter: test {len(test_oos)}, "
                f"S3 {len(s3_oos)}, S2 val {len(s2_oos)}. Used as many as available.")

        mixed = ([{"id": f"C-b{i}", "text": test[i][0], "gold": test[i][1], "source": "banking77_test",
                   "src_idx": i} for i in base]
                 + [{"id": f"C-o{i}", "text": clinc["test"][i]["text"], "gold": NONE_LABEL,
                     "source": "clinc_oos_test", "src_idx": i} for i in test_oos])
        random.Random(987).shuffle(mixed)
        s3_rows = [{"id": f"S3-{i}", "text": clinc["train"][i]["text"], "gold": NONE_LABEL,
                    "source": "clinc_oos_train", "src_idx": i} for i in s3_oos]
        s2_oos_rows = [{"id": f"V-o{sp}{i}", "text": clinc[sp][i]["text"], "gold": NONE_LABEL,
                        "source": f"clinc_oos_{sp}", "src_idx": i} for sp, i in s2_oos]
        val = {}
        for s in SEEDS:
            sub = set(pick_subset(train_all, 2000, s))
            cand = [i for i, ex in enumerate(train_all) if ex not in sub]
            pick = random.Random(555 + s).sample(cand, 500)
            val[s] = ([{"id": f"V{s}-b{i}", "text": train_all[i][0], "gold": train_all[i][1],
                        "source": "banking77_train", "src_idx": i} for i in pick] + s2_oos_rows)
        log["decisions"].append(
            "S2 validation in-scope items: 500 Banking77 train examples not in that seed's 2,000-example "
            "subset, drawn with Random(555 + seed).")
        log["counts"]["oos_mixed"] = len(mixed)

    # ------------------------------------------------------------- write (atomic)
    write_jsonl_atomic(os.path.join(d, "typo500.jsonl"), typo_rows)
    write_jsonl_atomic(os.path.join(d, "merged250.jsonl"), merged_rows)
    files = ["typo500.jsonl", "merged250.jsonl"]
    if c_ok:
        write_jsonl_atomic(os.path.join(d, "oos_mixed.jsonl"), mixed)
        write_jsonl_atomic(os.path.join(d, "s3_train_oos.jsonl"), s3_rows)
        for s in SEEDS:
            write_jsonl_atomic(os.path.join(d, f"s2_val_s{s}.jsonl"), val[s])
        files += ["oos_mixed.jsonl", "s3_train_oos.jsonl"] + [f"s2_val_s{s}.jsonl" for s in SEEDS]
    else:
        with open(os.path.join(d, "test_c_unavailable"), "w") as f:
            f.write(c_reason)
        log["decisions"].append(f"Test C skipped: {c_reason}")
    log["test_c_available"] = c_ok
    write_json_atomic(os.path.join(d, "build_log.json"), log)
    files.append("build_log.json")
    print(json.dumps(log["counts"], indent=1))
    for x in log["decisions"]:
        print("DECISION:", x)

    if a.commit:
        paths = [os.path.join(d, f) for f in files]
        subprocess.run(["git", "add", "-f"] + paths, check=True)
        r = subprocess.run(["git", "commit", "-q", "-m",
                            "Freeze T1 robustness test sets (built before any model output)"],
                           capture_output=True, text=True)
        print("git commit:", r.returncode, r.stdout.strip(), r.stderr.strip())


if __name__ == "__main__":
    main()
