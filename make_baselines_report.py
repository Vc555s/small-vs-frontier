#!/usr/bin/env python3
"""T1 baselines report (E1-E6): writes results/t1_baselines.md, <next>/summary.md,
<next>/joint_errors_for_review.csv and <next>/claude_calls_next.json. Every number is computed
from result files. Missing runs are reported as missing.

  python make_baselines_report.py [--next results/next] [--out results/t1_baselines.md]
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import statistics as st
from collections import Counter

from t1_common import load_data, system_prompt

TAG = "Qwen3-4B-Instruct-2507"
E4A, E4B = "bge-small-en-v1.5-logreg", "ModernBERT-base-ft"
BUDGETS = [100, 500, 2000, 9993]
SEEDS = [0, 1, 2]
BOOT_N, BOOT_SEED = 10000, 99
CAP = 140
CONTAMINATION = ("Banking77 has been public since 2020, so it may appear in the training data of any of "
                 "the models compared. This study cannot rule that out.")


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def bootstrap_diff(a, b):
    """95% paired bootstrap interval for mean(a) - mean(b); a, b are 0/1 lists (same items)."""
    rng = random.Random(BOOT_SEED)
    n = len(a)
    diffs = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * BOOT_N)], diffs[int(0.975 * BOOT_N) - 1]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def macro_f1(golds, preds, labels):
    from sklearn.metrics import f1_score
    return f1_score(golds, [p if p is not None else "__invalid__" for p in preds], labels=labels,
                    average="macro", zero_division=0)


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def ms(xs, d=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return f"{xs[0]:.{d}f}" if len(xs) == 1 else f"{st.mean(xs):.{d}f} ± {st.stdev(xs):.{d}f}"


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


# ------------------------------------------------------------------ loading
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next", default="results/next")
    ap.add_argument("--out", default="results/t1_baselines.md")
    a = ap.parse_args()
    N = a.next
    _, test, labels = load_data("mteb/banking77")
    idx = json.load(open("results/sample500.json"))["indices"]
    gold = {i: test[i][1] for i in idx}

    # per-run predictions on the 500 sample, keyed by test index -> pred
    runs = {}      # name -> dict(system, budget, seed, kind, preds{idx: pred}, full_acc, full_f1)

    def add_full_test(name, system, budget, seed, path, kind):
        if not os.path.exists(path):
            return
        rows = jl(path)
        if len(rows) != len(test) or any(rows[i]["text"] != test[i][0] for i in range(len(test))):
            return  # test-run files cover only part of the test set
        p = {i: rows[i]["pred"] for i in range(len(test))}
        runs[name] = {"system": system, "budget": budget, "seed": seed, "kind": kind,
                      "preds": {i: p[i] for i in idx},
                      "full_acc": sum(p[i] == test[i][1] for i in range(len(test))) / len(test),
                      "full_f1": macro_f1([t[1] for t in test], [p[i] for i in range(len(test))], labels)}

    def add_claude(name, system, path, kind, key="idx"):
        if not os.path.exists(path):
            return
        rows = jl(path)
        p = {r.get(key, r.get("src_idx")): r["pred"] for r in rows}
        if not all(i in p for i in idx):
            runs[name] = {"system": system, "kind": kind, "incomplete": f"{len(p)} of {len(idx)} messages",
                          "budget": None, "seed": None}
            return
        runs[name] = {"system": system, "budget": None, "seed": None, "kind": kind,
                      "preds": {i: p[i] for i in idx}}

    add_claude("claude_opus_zeroshot", "Claude Opus 5.5, zero-shot (T1)", "results/preds_claude_code_sample500.jsonl", "claude")
    add_claude("claude_opus_E2a", "Claude Opus 5.5, one example per intent (E2a)", f"{N}/preds_claude_E2a.jsonl", "claude", "src_idx")
    add_claude("claude_opus_E2b", "Claude Opus 5.5, retrieved examples (E2b)", f"{N}/preds_claude_E2b.jsonl", "claude", "src_idx")
    add_claude("claude_haiku_E3a", "Claude Haiku 5.5, zero-shot (E3a)", f"{N}/preds_claude_E3a.jsonl", "claude", "src_idx")
    add_claude("claude_haiku_E3b", "Claude Haiku 5.5, one example per intent (E3b)", f"{N}/preds_claude_E3b.jsonl", "claude", "src_idx")
    for n in BUDGETS:
        for s in SEEDS:
            qp = f"results/preds_{TAG}_n{n}_s{s}.jsonl"
            if not os.path.exists(qp):
                qp = f"{N}/preds_{TAG}_n{n}_s{s}.jsonl"
            add_full_test(f"qwen_n{n}_s{s}", "Qwen3-4B + LoRA", n, s, qp, "qwen")
            add_full_test(f"e4a_n{n}_s{s}", "bge-small + logistic regression (E4a)", n, s, f"{N}/preds_{E4A}_n{n}_s{s}.jsonl", "e4a")
            add_full_test(f"e4b_n{n}_s{s}", "ModernBERT-base fine-tuned (E4b)", n, s, f"{N}/preds_{E4B}_n{n}_s{s}.jsonl", "e4b")

    def correct(name):
        r = runs[name]
        return [int(r["preds"][i] == gold[i]) for i in idx]

    ref = "claude_opus_zeroshot"
    c_ref = correct(ref)
    complete = [k for k, r in runs.items() if "preds" in r]

    # comparisons against Claude zero-shot (Holm family = every run below)
    comp = []
    for k in complete:
        if k == ref:
            continue
        x = correct(k)
        b = sum(1 for u, v in zip(x, c_ref) if u and not v)
        c = sum(1 for u, v in zip(x, c_ref) if v and not u)
        lo, hi = bootstrap_diff(x, c_ref)
        comp.append({"run": k, "diff": (sum(x) - sum(c_ref)) / len(idx), "lo": lo, "hi": hi, "b": b, "c": c,
                     "p": mcnemar(b, c)})
    adj = holm([r["p"] for r in comp])
    for r, h in zip(comp, adj):
        r["p_holm"] = h
    cmap = {r["run"]: r for r in comp}

    # ------------------------------------------------------------------ costs and latency
    calls = []
    for f in sorted(glob.glob(f"{N}/claude_raw/*/*.json")):
        raw = json.load(open(f))
        m = raw.get("stdout_json") or {}
        u = m.get("usage") or {}
        calls.append({"set": raw.get("set"), "batch": raw.get("batch"), "attempt": raw.get("attempt"),
                      "cli_version": raw.get("cli_version"), "returncode": raw.get("returncode"),
                      "wall_seconds": raw.get("wall_seconds"), "num_turns": m.get("num_turns"),
                      "prompt_tokens": (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                      + (u.get("cache_read_input_tokens") or 0),
                      "output_tokens": u.get("output_tokens"), "cli_list_price_estimate_usd": m.get("total_cost_usd"),
                      "models": sorted((m.get("modelUsage") or {}).keys())})
    if calls:
        json.dump({"note": "Per-call summary of the Claude Code CLI output for E2 and E3, identifiers removed "
                           "(no session ids, paths or prompts). The cost is the CLI's own list-price estimate.",
                   "calls": calls}, open(f"{N}/claude_calls_next.json", "w"), indent=1)
    elif os.path.exists(f"{N}/claude_calls_next.json"):
        calls = json.load(open(f"{N}/claude_calls_next.json"))["calls"]
    calls_used = sum(1 for _ in calls)
    t1_calls = json.load(open("results/claude_calls_t1.json"))["calls"]

    def claude_cost(set_name):
        cs = [c for c in calls if c["set"] == set_name] if set_name else \
             [{"cli_list_price_estimate_usd": c["cli_list_price_estimate_usd"], "prompt_tokens":
               c["usage"]["input_tokens"] + c["usage"]["cache_creation_input_tokens"] + c["usage"]["cache_read_input_tokens"]}
              for c in t1_calls]
        if not cs:
            return None, None
        usd = sum(c["cli_list_price_estimate_usd"] or 0 for c in cs) / 500 * 1000
        tok = st.mean(c["prompt_tokens"] for c in cs)
        return usd, tok

    cost_rows = {"claude_opus_zeroshot": claude_cost(None), "claude_opus_E2a": claude_cost("E2a"),
                 "claude_opus_E2b": claude_cost("E2b"), "claude_haiku_E3a": claude_cost("E3a"),
                 "claude_haiku_E3b": claude_cost("E3b")}
    import compute_cost
    lat = {"qwen": "results/latency_t1.json", "e4a": f"{N}/latency_e4a.json", "e4b": f"{N}/latency_e4b.json"}
    lat = {k: json.load(open(v)) for k, v in lat.items() if os.path.exists(v)}
    rates = compute_cost.RENTAL_EUR_PER_GPU_HOUR

    def gpu_cost(kind):
        if kind not in lat:
            return None
        mps = lat[kind]["throughput"]["16"]["messages_per_second"]
        return [1000 / mps / 3600 * r for r in rates]

    # ------------------------------------------------------------------ report
    L = []
    w = L.append
    w("# T1 baselines: examples in the prompt, cheaper Claude models, non-generative models, more seeds")
    w("")
    w("Generated by `make_baselines_report.py` from the files in `results/` and `results/next/`. Every number "
      "is computed from those files. All systems are scored on the same 500 Banking77 test messages "
      "(`results/sample500.json`) with exact match after the same normalisation.")
    w("")
    missing = [k for k, r in runs.items() if "incomplete" in r]
    expected = (["claude_opus_E2a", "claude_opus_E2b", "claude_haiku_E3a", "claude_haiku_E3b"]
                + [f"qwen_n9993_s{s}" for s in SEEDS]
                + [f"{p}_n{n}_s{s}" for p in ("e4a", "e4b") for n in BUDGETS for s in SEEDS])
    absent = [k for k in expected if k not in runs]
    if missing or absent:
        w("## Not completed")
        w("")
        for k in missing:
            w(f"- `{k}`: incomplete ({runs[k]['incomplete']})")
        for k in absent:
            w(f"- `{k}`: no result file")
        w("")

    # main table
    w("## 1. All systems on the 500-message sample")
    w("")
    w("Multi-seed systems: accuracy and macro-F1 are mean ± std over seeds; the Wilson interval, the "
      "difference against Claude zero-shot (with its 95% paired bootstrap interval, 10,000 resamples, "
      "`random.Random(99)`) and the exact McNemar p-values are for seed 0. Holm-corrected p-values use every "
      "run compared with Claude zero-shot in this report as one family "
      f"({len(comp)} comparisons; all seeds are in section 2).")
    w("")
    w("| System | Labels | Runs | Accuracy | Macro-F1 | Valid labels | Wilson 95% (seed 0) | Δ vs Claude zero-shot (95% bootstrap) | p raw | p Holm | Batch-1 p50 | Cost / 1,000 requests |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")

    def row_for(keys, system, labels_txt, kind):
        keys = [k for k in keys if k in runs and "preds" in runs[k]]
        if not keys:
            w(f"| {system} | {labels_txt} | 0 | not completed | | | | | | | | |")
            return
        accs = [sum(correct(k)) / len(idx) for k in keys]
        f1s = [macro_f1([gold[i] for i in idx], [runs[k]["preds"][i] for i in idx], labels) for k in keys]
        k0 = keys[0]
        vals = [sum(runs[k]["preds"][i] is not None for i in idx) / len(idx) for k in keys]
        lo, hi = wilson(sum(correct(k0)), len(idx))
        if k0 == ref:
            dtxt, praw, pholm = "reference", "", ""
        else:
            c = cmap[k0]
            dtxt = f"{100 * c['diff']:+.1f} pts ({100 * c['lo']:+.1f} to {100 * c['hi']:+.1f})"
            praw, pholm = f"{c['p']:.3g}", f"{c['p_holm']:.3g}"
        if kind == "claude":
            usd, tok = cost_rows.get(k0, (None, None))
            lat_txt = "n/a (CLI wall time is not latency)"
            cost_txt = f"USD {usd:.2f} (CLI estimate; {tok:,.0f} prompt tokens per call)" if usd is not None else "n/a"
        else:
            lk = lat.get(kind)
            lat_txt = f"{lk['batch1_latency_ms']['p50']} ms" if lk else "n/a"
            gc_ = gpu_cost(kind)
            cost_txt = (" / ".join(f"EUR {x:.4f}" for x in gc_) + f" at EUR {' / '.join(f'{r:.2f}' for r in rates)} per GPU-hour"
                        if gc_ else "n/a")
        w(f"| {system} | {labels_txt} | {len(keys)} | {ms(accs)} | {ms(f1s)} | {ms(vals)} | {lo:.3f}–{hi:.3f} | {dtxt} | "
          f"{praw} | {pholm} | {lat_txt} | {cost_txt} |")

    row_for([ref], "Claude Opus 5.5, zero-shot", "0", "claude")
    row_for(["claude_opus_E2a"], "Claude Opus 5.5, one example per intent (E2a)", "77 in prompt", "claude")
    row_for(["claude_opus_E2b"], "Claude Opus 5.5, retrieved examples (E2b)", "≤125 in prompt", "claude")
    row_for(["claude_haiku_E3a"], "Claude Haiku 5.5, zero-shot (E3a)", "0", "claude")
    row_for(["claude_haiku_E3b"], "Claude Haiku 5.5, one example per intent (E3b)", "77 in prompt", "claude")
    for n in BUDGETS:
        row_for([f"qwen_n{n}_s{s}" for s in SEEDS], "Qwen3-4B + LoRA", f"{n:,}", "qwen")
        row_for([f"e4b_n{n}_s{s}" for s in SEEDS], "ModernBERT-base fine-tuned (E4b)", f"{n:,}", "e4b")
        row_for([f"e4a_n{n}_s{s}" for s in SEEDS], "bge-small + logistic regression (E4a)", f"{n:,}", "e4a")
    w("")
    w("Qwen latency is the merged out-of-scope adapter from `results/latency_t1.json` (same base model and "
      "LoRA settings); E4a and E4b latency is measured on the 2,000-label seed-0 model. Latency was measured on a "
      "shared GPU and is not best-case. GPU cost per 1,000 requests = 1,000 ÷ throughput at batch 16 ÷ 3,600 × the "
      "assumed hourly rate (`compute_cost.py`); the rates are assumptions, not quoted prices. Claude cost is the "
      "Claude Code CLI's own list-price estimate, scaled from 500 to 1,000 messages; it is not a measured API bill "
      "and not a charge on a subscription.")
    w("")

    # per-run comparisons
    w("## 2. Every run against Claude zero-shot (exact McNemar, Holm-corrected)")
    w("")
    w("| Run | Accuracy | Δ (95% bootstrap) | Run right, Claude wrong | Claude right, run wrong | p raw | p Holm |")
    w("|---|---|---|---|---|---|---|")
    def run_key(r):
        k = r["run"]
        if k.startswith("claude"):
            return (0, k, 0, 0)
        p, n, s = k.split("_")
        return ({"qwen": 1, "e4b": 2, "e4a": 3}[p], "", int(n[1:]), int(s[1:]))
    for c in sorted(comp, key=run_key):
        w(f"| {c['run']} | {sum(correct(c['run'])) / len(idx):.3f} | {100 * c['diff']:+.1f} pts "
          f"({100 * c['lo']:+.1f} to {100 * c['hi']:+.1f}) | {c['b']} | {c['c']} | {c['p']:.3g} | {c['p_holm']:.3g} |")
    w("")
    nsig = [c for c in comp if c["run"].startswith("qwen_n9993") and c["p_holm"] >= 0.05]
    q9 = [c for c in comp if c["run"].startswith("qwen_n9993")]
    if q9:
        w("E1, full training set: " + "; ".join(
            f"seed {c['run'][-1]} {100 * c['diff']:+.1f} pts (Holm p = {c['p_holm']:.3g})" for c in sorted(q9, key=lambda r: r["run"])) + ". "
          + ("Every seed's lead over Claude is significant after correction." if not nsig and len(q9) == 3 else
             f"Not significant after correction for: {', '.join(c['run'] for c in nsig) or 'none'}."))
        w("")

    # E4 vs Qwen
    w("## 3. Non-generative baselines against Qwen with the same budget and seed (exact McNemar, uncorrected)")
    w("")
    w("| Run | Qwen same budget and seed | Run right, Qwen wrong | Qwen right, run wrong | p |")
    w("|---|---|---|---|---|")
    for p in ("e4a", "e4b"):
        for n in BUDGETS:
            for s in SEEDS:
                k, q = f"{p}_n{n}_s{s}", f"qwen_n{n}_s{s}"
                if k in complete and q in complete:
                    x, y = correct(k), correct(q)
                    b = sum(1 for u, v in zip(x, y) if u and not v)
                    c = sum(1 for u, v in zip(x, y) if v and not u)
                    w(f"| {k} | {sum(y) / len(idx):.3f} | {b} | {c} | {mcnemar(b, c):.3g} |")
    w("")

    # label efficiency
    w("## 4. Label efficiency (mean ± std over seeds)")
    w("")
    w("At 9,993 labels every seed uses the same full training set, so seeds differ only in initialisation and "
      "data order (for E4a, which has no random initialisation, only the order of the training rows differs between "
      "seeds, which changes the solver's numerical path slightly).")
    w("")
    w("| Labels | Qwen3-4B + LoRA (500 / full test) | ModernBERT-base (500 / full test) | bge-small + LR (500 / full test) |")
    w("|---|---|---|---|")
    for n in BUDGETS:
        cells = []
        for p in ("qwen", "e4b", "e4a"):
            ks = [f"{p}_n{n}_s{s}" for s in SEEDS if f"{p}_n{n}_s{s}" in complete]
            cells.append(f"{ms([sum(correct(k)) / len(idx) for k in ks])} / {ms([runs[k]['full_acc'] for k in ks], 4)}"
                         if ks else "n/a")
        w(f"| {n:,} | " + " | ".join(cells) + " |")
    w("")

    # E5 error analysis
    w("## 5. Error analysis")
    w("")
    ea = [("Claude zero-shot", ref), ("Claude E2a", "claude_opus_E2a"), ("Qwen 2,000 s0", "qwen_n2000_s0"),
          ("Qwen 9,993 s0", "qwen_n9993_s0")]
    ea = [(n_, k) for n_, k in ea if k in complete]
    for name, k in ea:
        conf = Counter((gold[i], runs[k]["preds"][i]) for i in idx if runs[k]["preds"][i] != gold[i])
        w(f"### {name}: 15 most frequent confusions")
        w("")
        w("| Gold | Predicted | Count |")
        w("|---|---|---|")
        for (g, pr), c in conf.most_common(15):
            w(f"| {g} | {pr} | {c} |")
        w("")
        per = {}
        for i in idx:
            per.setdefault(gold[i], []).append(runs[k]["preds"][i] == gold[i])
        worst = sorted(per.items(), key=lambda kv: (sum(kv[1]) / len(kv[1]), kv[0]))[:10]
        w(f"Lowest per-intent accuracy (500 sample; few messages per intent): "
          + "; ".join(f"{g} {sum(v)}/{len(v)}" for g, v in worst))
        w("")
    if len(ea) >= 2:
        w("### Error overlap (500 sample)")
        w("")
        w("| A | B | Both wrong | Only A wrong | Only B wrong |")
        w("|---|---|---|---|---|")
        for x in range(len(ea)):
            for y in range(x + 1, len(ea)):
                (na, ka), (nb, kb) = ea[x], ea[y]
                ca, cb = correct(ka), correct(kb)
                w(f"| {na} | {nb} | {sum(1 for u, v in zip(ca, cb) if not u and not v)} | "
                  f"{sum(1 for u, v in zip(ca, cb) if not u and v)} | {sum(1 for u, v in zip(ca, cb) if u and not v)} |")
        w("")
    if "qwen_n9993_s0" in complete:
        both = [i for i in idx if runs["qwen_n9993_s0"]["preds"][i] != gold[i] and runs[ref]["preds"][i] != gold[i]]
        with open(f"{N}/joint_errors_for_review.csv", "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["test_idx", "text", "gold", "qwen_9993_s0_pred", "claude_zeroshot_pred", "judgement"])
            for i in both:
                cw.writerow([i, test[i][0], gold[i], runs["qwen_n9993_s0"]["preds"][i], runs[ref]["preds"][i], ""])
        w(f"`results/next/joint_errors_for_review.csv` lists the {len(both)} messages that both Qwen 9,993 (seed 0) "
          "and Claude zero-shot got wrong, with an empty `judgement` column (`label_wrong`, `ambiguous`, "
          "`models_wrong`) for manual review.")
        w("")

    # overlap between test and training messages
    from t1_common import load_data as _ld, normalised_overlap
    train_all, _, _ = _ld("mteb/banking77")
    n_exact, overlap = normalised_overlap(train_all, test)
    ov_idx = {i for i, _, _ in overlap}
    w("## 6. Test messages that also appear in training")
    w("")
    w(f"Exact-text overlap between the test and training splits is {n_exact}. After lowercasing and trimming "
      f"surrounding whitespace, {len(overlap)} test messages also occur in the training split "
      f"({sum(i in set(idx) for i in ov_idx)} of them in the 500-message sample). The same overlap exists in the "
      "original `PolyAI/banking77` splits; it is not introduced by the `mteb/banking77` copy.")
    w("")
    w("| Test idx | Label | In 500 sample | Message |")
    w("|---|---|---|---|")
    for i, t_, l_ in overlap:
        w(f"| {i} | {l_} | {'yes' if i in set(idx) else 'no'} | {esc(' '.join(t_.split()))} |")
    w("")
    w("Accuracy with and without these messages (the main tables above are unchanged):")
    w("")
    w("| System | Runs | 500 sample, all (n=500) | 500 sample, without overlap | Full test, all (n=3,076) | Full test, without overlap |")
    w("|---|---|---|---|---|---|")
    keep500 = [i for i in idx if i not in ov_idx]
    keepall = [i for i in range(len(test)) if i not in ov_idx]

    def full_preds(path):
        if not os.path.exists(path):
            return None
        rows = jl(path)
        return [r["pred"] for r in rows] if len(rows) == len(test) else None

    def ov_row(name, keys, full_paths):
        keys = [k for k in keys if k in complete]
        if not keys:
            w(f"| {name} | 0 | not completed | | | |")
            return
        a5 = [sum(runs[k]["preds"][i] == gold[i] for i in idx) / len(idx) for k in keys]
        b5 = [sum(runs[k]["preds"][i] == gold[i] for i in keep500) / len(keep500) for k in keys]
        fa, fb = [], []
        for fp in full_paths:
            pr = full_preds(fp)
            if pr:
                fa.append(sum(pr[i] == test[i][1] for i in range(len(test))) / len(test))
                fb.append(sum(pr[i] == test[i][1] for i in keepall) / len(keepall))
        w(f"| {name} | {len(keys)} | {ms(a5)} | {ms(b5)} (n={len(keep500)}) | {ms(fa) if fa else 'n/a'} | "
          f"{(ms(fb) + f' (n={len(keepall):,})') if fb else 'n/a'} |")

    def qp(n, s):
        q = f"results/preds_{TAG}_n{n}_s{s}.jsonl"
        return q if os.path.exists(q) else f"{N}/preds_{TAG}_n{n}_s{s}.jsonl"

    ov_row("Claude Opus 5.5, zero-shot", [ref], [])
    ov_row("Claude Opus 5.5, retrieved examples (E2b)", ["claude_opus_E2b"], [])
    ov_row("Qwen3-4B + LoRA, 2,000 labels", [f"qwen_n2000_s{s}" for s in SEEDS], [qp(2000, s) for s in SEEDS])
    ov_row("Qwen3-4B + LoRA, 9,993 labels", [f"qwen_n9993_s{s}" for s in SEEDS], [qp(9993, s) for s in SEEDS])
    ov_row("ModernBERT-base fine-tuned (E4b), 9,993 labels", [f"e4b_n9993_s{s}" for s in SEEDS],
           [f"{N}/preds_{E4B}_n9993_s{s}.jsonl" for s in SEEDS])
    ov_row("bge-small + logistic regression (E4a), 9,993 labels", [f"e4a_n9993_s{s}" for s in SEEDS],
           [f"{N}/preds_{E4A}_n9993_s{s}.jsonl" for s in SEEDS])
    w("")
    w("Claude was run on the 500 sample only, so it has no full-test figures.")
    w("")

    # prompts
    w("## 7. Prompts and example selection")
    w("")
    w("All Claude runs use the T1 instructions (`system_prompt(labels)`), the T1 batch order "
      "(`results/sample500_batch_order.json`, 25 messages per call) and the T1 batch instruction. Example "
      "variants add a block headed `Examples:` with one line per example in the form `<message> => <label>`, "
      "placed between the instructions and the batch instruction.")
    w("")
    w("- **E2a / E3b:** one example per intent (77): the seed-0 2,000-example training subset shuffled with "
      "`random.Random(2024)`, first example of each intent, kept in that order (`results/next/icl_one_per_class.json`). "
      "The same 77 examples go into every call.")
    w("- **E2b:** for each message in a call, its 5 nearest examples in the seed-0 2,000-example subset by "
      "cosine similarity of `BAAI/bge-small-en-v1.5` embeddings (no query instruction); the union per call, "
      "duplicates removed, sorted by subset position and shuffled with `random.Random(7)` "
      "(`results/next/icl_retrieved/batch_XX.json`).")
    w("- **E3a / E3b:** as zero-shot / E2a, with `--model claude-haiku-5-5`.")
    w("")
    one = json.load(open(f"{N}/icl_one_per_class.json"))["examples"] if os.path.exists(f"{N}/icl_one_per_class.json") else []
    if one:
        from baseline_claude_code import BATCH_ADDITION
        prompt = (system_prompt(labels) + "\n\nExamples:\n"
                  + "\n".join(f"{' '.join(e['text'].split())} => {e['label']}" for e in one) + BATCH_ADDITION)
        w("<details><summary>Exact E2a / E3b system prompt</summary>")
        w("")
        w("```")
        w(prompt)
        w("```")
        w("")
        w("</details>")
        w("")

    # Claude call facts
    w("## 8. Claude calls")
    w("")
    if calls:
        by = Counter(c["set"] for c in calls)
        w(f"Calls used in this task: {calls_used} of {CAP} (includes test-run calls kept as batch 0 and retries). "
          "Per set: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())) + ". CLI version "
          + ", ".join(sorted({str(c['cli_version']) for c in calls})) + ". Answering models: "
          + ", ".join(sorted({m for c in calls for m in c["models"]})) + ".")
        w("")
        w("| Set | Calls | Retry calls | Messages without an answer | Valid labels | Mean prompt tokens | Mean wall s | CLI estimate per 1,000 messages |")
        w("|---|---|---|---|---|---|---|---|")
        unanswered = {}
        for s_ in sorted(by):
            cs = [c for c in calls if c["set"] == s_]
            pp = f"{N}/preds_claude_{s_}.jsonl"
            rows_ = jl(pp) if os.path.exists(pp) else []
            unanswered[s_] = sum(1 for r in rows_ if r.get("error"))
            nvalid = sum(1 for r in rows_ if r["pred"] is not None)
            w(f"| {s_} | {len(cs)} | {sum(1 for c in cs if c['attempt'] > 1)} | {unanswered[s_]} | {nvalid}/{len(rows_)} | "
              f"{st.mean(c['prompt_tokens'] for c in cs):,.0f} | {st.mean(c['wall_seconds'] for c in cs):.1f} | "
              f"USD {sum(c['cli_list_price_estimate_usd'] or 0 for c in cs) / 500 * 1000:.2f} |")
        w("")
        w("A reply counts as malformed when it is not a JSON array of exactly 25 strings (for example 24 or 26 labels). "
          "A malformed batch is retried once with the identical prompt; if the retry is also malformed, its 25 messages "
          "get no answer and **count as wrong**, as fixed before the runs. Accuracy on the answered messages only, for "
          "reference: " + "; ".join(
              f"{s_} {sum(r['pred'] == r['gold'] for r in rs_) / len(rs_):.3f} on {len(rs_)}"
              for s_ in sorted(by)
              for rs_ in [[r for r in jl(f'{N}/preds_claude_{s_}.jsonl') if not r.get('error')]] if rs_) + ".")
        w("")
    else:
        w("No calls recorded.")
        w("")

    # caveats
    w("## 9. Caveats")
    w("")
    for c in [
        CONTAMINATION,
        f"Exact-text overlap between the test and training splits is {n_exact}; after lowercasing and trimming "
        f"whitespace it is {len(overlap)} messages, and the same overlap exists in the original Banking77 splits. "
        "Section 6 gives the accuracies with these messages removed.",
        "Claude saw 25 messages per call; the small models saw one at a time.",
        "The Claude Code CLI adds some prompt text of its own; its temperature cannot be set; each Claude "
        "variant was run once. These are calls through one CLI on a subscription, not API baselines.",
        "The zero-shot Opus run (T1) used CLI version 2.1.322 and was launched from a shell with Claude Code "
        "session environment variables set; the E2 and E3 calls used CLI version 2.1.323 from a clean environment.",
        "Example prompts are longer, so they cost more per request; the cost column uses the CLI's own estimate.",
        "Latency was measured on a shared GPU with other users' jobs running; the numbers are not best-case.",
        "At 9,993 labels the three seeds share the same training set.",
    ]:
        w(f"- {c}")
    w("")
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", a.out)

    # ------------------------------------------------------------------ summary.md
    S = ["# Next experiments: summary", ""]

    def acc_of(k):
        return sum(correct(k)) / len(idx) if k in complete else None

    unans = {}
    for s_ in ("E2a", "E2b", "E3a", "E3b"):
        pp = f"{N}/preds_claude_{s_}.jsonl"
        if os.path.exists(pp):
            rs_ = jl(pp)
            ans = [r for r in rs_ if not r.get("error")]
            unans[f"claude_{'opus' if s_.startswith('E2') else 'haiku'}_{s_}"] = (len(rs_) - len(ans),
                sum(r["pred"] == r["gold"] for r in ans) / len(ans) if ans else None, len(ans))

    def line(label, k):
        if k not in complete:
            return f"- {label}: not completed."
        extra = ""
        if unans.get(k, (0,))[0]:
            u, a_acc, n_ans = unans[k]
            extra = (f" {u} of 500 messages got no answer because the reply was malformed twice (wrong number of "
                     f"labels); they count as wrong. On the {n_ans} answered messages its accuracy is {a_acc:.3f}.")
        c = cmap.get(k)
        if not c:
            return f"- {label}: {acc_of(k):.3f}.{extra}"
        return (f"- {label}: {acc_of(k):.3f} ({100 * c['diff']:+.1f} points against Claude zero-shot "
                f"{acc_of(ref):.3f}; Holm-corrected p = {c['p_holm']:.3g}).{extra}")
    S += ["## Results (500-message sample)", ""]
    S.append(line("Claude Opus with one example per intent (E2a)", "claude_opus_E2a"))
    S.append(line("Claude Opus with retrieved examples (E2b)", "claude_opus_E2b"))
    S.append(line("Claude Haiku zero-shot (E3a)", "claude_haiku_E3a"))
    S.append(line("Claude Haiku with one example per intent (E3b)", "claude_haiku_E3b"))
    for s in SEEDS:
        S.append(line(f"Qwen3-4B + LoRA, 9,993 labels, seed {s} (E1)", f"qwen_n9993_s{s}"))
    for p, nm in (("e4b", "ModernBERT-base fine-tuned (E4b)"), ("e4a", "bge-small + logistic regression (E4a)")):
        for n in BUDGETS:
            ks = [f"{p}_n{n}_s{s}" for s in SEEDS if f"{p}_n{n}_s{s}" in complete]
            if ks:
                S.append(f"- {nm}, {n:,} labels: {ms([acc_of(k) for k in ks])} over {len(ks)} seed(s).")
    S += ["", "Full tables, paired tests, error analysis and prompts: `results/t1_baselines.md`.", "",
          "## Claude calls", "", f"{calls_used} of {CAP} used (E2c and the Sonnet runs were not run).", "",
          "## Deviations", ""]
    dev_path = f"{N}/deviations.json"
    devs = json.load(open(dev_path)) if os.path.exists(dev_path) else []
    S += [f"- {d}" for d in devs] or ["- None recorded."]
    if missing or absent:
        S += ["", "## Not completed", ""] + [f"- {k}" for k in missing + absent]
    with open(f"{N}/summary.md", "w") as f:
        f.write("\n".join(S) + "\n")
    print("wrote", f"{N}/summary.md")


if __name__ == "__main__":
    main()
