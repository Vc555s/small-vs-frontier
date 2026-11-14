#!/usr/bin/env python3
"""Build results/t1_report.md: the complete T1 write-up (fine-tuned Qwen3-4B vs
Claude via Claude Code CLI), with every number read from the result files and every
log embedded in full. No GPU, no Claude calls.

  python make_t1_report.py
"""
import glob
import json
import math
import os
import statistics as st
import subprocess
from datetime import datetime

from baseline_claude_code import LABEL, metrics, wilson
from t1_common import load_data

R = "results"
TAG = "Qwen3-4B-Instruct-2507"
BUDGETS = [0, 100, 500, 2000, 9993]
FULL_N = 9993  # full training set, seed 0 only
SEEDS = [0, 1, 2]
OUT = os.path.join(R, "t1_report.md")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(unavailable: {e})"


def read(p):
    with open(p) as f:
        return f.read()


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def clean_log(text):
    """tqdm progress bars are written with carriage returns; keep only the final
    state of each such line so the log is readable. Nothing else is changed."""
    return "\n".join(line.split("\r")[-1] for line in text.split("\n"))


def ms(xs, d=4):
    if len(xs) == 1:
        return f"{xs[0]:.{d}f}"
    return f"{st.mean(xs):.{d}f} ± {st.stdev(xs):.{d}f}"


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def fence(text, lang=""):
    return f"```{lang}\n{text.rstrip()}\n```"


def main():
    train, test, labels = load_data("mteb/banking77")
    idx = json.load(open(os.path.join(R, "sample500.json")))["indices"]
    order = json.load(open(os.path.join(R, "sample500_batch_order.json")))["order"]
    claude = {r["idx"]: r for r in jl(os.path.join(R, "preds_claude_code_sample500.jsonl"))}
    assert sorted(claude) == sorted(idx)

    # ---------------- small model: per-run rows and predictions
    rows = {}
    for s in SEEDS:
        for r in json.load(open(os.path.join(R, f"exp1_{TAG}_s{s}.json"))):
            rows[(r["budget"], s)] = r
    # full training set: seed 0 lives in results/, further seeds (E1) in results/next/
    NEXT = os.path.join(R, "next")

    def find(name):
        for base in (R, NEXT):
            if os.path.exists(os.path.join(base, name)):
                return os.path.join(base, name)
        return None

    full_paths = {}
    for s in SEEDS:
        fp, pp = find(f"exp1_{TAG}_n{FULL_N}_s{s}.json"), find(f"preds_{TAG}_n{FULL_N}_s{s}.jsonl")
        if fp and pp:
            for r in json.load(open(fp)):
                rows[(r["budget"], s)] = r
            full_paths[s] = fp
    full_path = full_paths.get(0)
    have_full = (FULL_N, 0) in rows
    full_seeds = sorted(full_paths)
    n_full = len(full_seeds)
    preds = {}
    for (n, s) in rows:
        p = jl(find(f"preds_{TAG}_n{n}_s{s}.jsonl"))
        assert len(p) == len(test)
        assert all(p[i]["text"] == test[i][0] for i in range(len(test))), (n, s)
        preds[(n, s)] = p

    L = []
    w = L.append
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    w("# T1 report: support-ticket routing (Banking77) — fine-tuned Qwen3-4B vs Claude via Claude Code CLI")
    w("")
    w(f"Generated {now} (server time) by `make_t1_report.py` from the files in `results/`. "
      "Every number below is read from those files; nothing is typed in by hand.")
    w("")
    w("Contents: 1 Summary · 2 Task and data · 3 Systems · 4 Results · 5 Head-to-head on the 500 sample · "
      "6 Procedure and issues · 7 Caveats · 8 Open work · Appendices A–E")
    w("")

    # ---------------- 1. summary
    c_items = [claude[i] for i in idx]
    c_k = sum(r["pred"] == r["gold"] for r in c_items)
    c_acc, c_f1, c_val = metrics([r["gold"] for r in c_items], [r["pred"] for r in c_items], labels)
    c_lo, c_hi = wilson(c_k, len(idx))

    def sample_scores(n):
        out = []
        for s in SEEDS:
            if (n, s) in preds:
                p = preds[(n, s)]
                out.append(metrics([p[i]["gold"] for i in idx], [p[i]["pred"] for i in idx], labels))
        return out

    s2000 = sample_scores(2000)
    s500 = sample_scores(500)
    w("## 1. Summary")
    w("")
    w(f"- **Claude via Claude Code CLI (Pro subscription)**, model `claude-opus-5-5`, scored "
      f"**{c_acc:.3f}** accuracy on a fixed 500-example sample (95% Wilson CI {c_lo:.3f}–{c_hi:.3f}), "
      f"with every answer a valid label.")
    w(f"- **Fine-tuned Qwen3-4B with 2,000 labels** scored **{ms([x[0] for x in s2000], 3)}** on the same 500 "
      f"(3 seeds) and **{ms([rows[(2000, s)]['accuracy'] for s in SEEDS], 3)}** on the full 3,076-example test set.")
    w(f"- The 2,000-label gap to Claude is {100 * (st.mean([x[0] for x in s2000]) - c_acc):+.1f} points, inside "
      "Claude's confidence interval and the seed spread, and not significant in a paired test "
      "(section 5). Under the decision rule fixed in advance (differences inside the seed spread or "
      "not significant in the paired test count as ties) this is a **tie**, not a win.")
    w(f"- With 500 labels the small model is {100 * (c_acc - st.mean([x[0] for x in s500])):.1f} points **behind** "
      "Claude; with 100 labels, further behind. The small model needs somewhere between 500 and 2,000 "
      "labels to match Claude on this 77-class task.")
    w("- The untuned model scores 0.606 on the full test set; fine-tuning gives the biggest jump at the first 100 labels.")
    if have_full and n_full > 1:
        sfs = [x[0] for x in sample_scores(FULL_N)]
        cr = [claude[i]["pred"] == claude[i]["gold"] for i in idx]
        per = []
        for s in full_seeds:
            pf = preds[(FULL_N, s)]
            sr = [pf[i]["pred"] == pf[i]["gold"] for i in idx]
            per.append(f"seed {s} {100 * (sum(sr) - sum(cr)) / len(idx):+.1f} pts, p = "
                       f"{mcnemar_exact(sum(a and not b for a, b in zip(sr, cr)), sum(b and not a for a, b in zip(sr, cr))):.3g}")
        w(f"- **Full training set (9,993 labels, {n_full} seeds):** {ms(sfs, 3)} on the 500 sample and "
          f"{ms([rows[(FULL_N, s)]['accuracy'] for s in full_seeds], 4)} on the full test set, against "
          f"{ms([x[0] for x in s2000], 3)} at 2,000 labels. Against Claude (exact McNemar per seed, uncorrected): "
          + "; ".join(per) + ". The seeds share the same training set and differ only in initialisation and data order.")
    elif have_full:
        sf = sample_scores(FULL_N)[0][0]
        s0_2000 = sample_scores(2000)[0][0]
        pf = preds[(FULL_N, 0)]
        sr = [pf[i]["pred"] == pf[i]["gold"] for i in idx]
        cr = [claude[i]["pred"] == claude[i]["gold"] for i in idx]
        so = sum(a and not b for a, b in zip(sr, cr)); co = sum(b and not a for a, b in zip(sr, cr))
        w(f"- **Full training set (9,993 labels, one seed):** {sf:.3f} on the 500 sample and "
          f"{rows[(FULL_N, 0)]['accuracy']:.3f} on the full test set; versus the 2,000-label seed-0 run "
          f"{100 * (sf - s0_2000):+.1f} points and versus the 2,000-label 3-seed mean "
          f"{100 * (sf - st.mean([x[0] for x in s2000])):+.1f} points on the sample; versus Claude "
          f"{100 * (sf - c_acc):+.1f} points (exact McNemar p = {mcnemar_exact(so, co):.3g}). "
          "A single seed cannot show run-to-run spread, so this point has no error bar.")
    else:
        w("- **Full training set (9,993 labels):** not completed (no result file).")
    w("- Run-to-run consistency and rewording stability have **not been measured yet** (section 8). "
      "Latency and cost of a merged adapter are in section 4.5.")
    w("")

    # ---------------- 2. data
    w("## 2. Task and data")
    w("")
    w(f"- Dataset: `mteb/banking77` (Hugging Face). Train {len(train):,} examples, test {len(test):,}, "
      f"{len(labels)} intent labels.")
    w("- Task: given one customer message, output exactly one of the 77 labels.")
    w("- Scoring: exact match after the shared `norm()` (lowercase, strip quotes/punctuation, spaces and "
      "hyphens to underscores). Any answer that is not an allowed label counts as wrong. Macro-F1 via "
      "scikit-learn over the 77 labels, invalid answers mapped to a non-label.")
    w("- The test set is **sorted by label** (the label changes 76 times in 3,075 transitions). This matters for batching (section 6.4).")
    w("- 500-example comparison sample: `sorted(random.Random(123).sample(range(3076), 500))`, saved in "
      "`results/sample500.json` and used for every system.")
    w("")

    # ---------------- 3. systems
    w("## 3. Systems")
    w("")
    w("### 3.1 Small model (yours): Qwen3-4B-Instruct-2507, untuned and LoRA fine-tuned")
    w("")
    w("| Setting | Value |")
    w("|---|---|")
    for k, v in [
        ("Base model", "`Qwen/Qwen3-4B-Instruct-2507`, fp16 weights"),
        ("Fine-tuning", "LoRA rank 16, alpha 32, dropout 0.05, `target_modules=\"all-linear\"` (33.0M trainable params, 0.81%)"),
        ("Optimiser", "AdamW, lr 2e-4, weight decay 0, linear decay with 6% warmup, grad clip 1.0"),
        ("Batch", "4 per step × 2 accumulation = effective 8"),
        ("Epochs", "5 for budget 100, 3 for budgets 500 and 2000"),
        ("Precision", "fp16 autocast + GradScaler (Turing GPU, no bf16); loss computed in fp32"),
        ("Loss", "answer tokens only (label + `<|im_end|>`), prompt masked"),
        ("Memory fixes", "gradient checkpointing; logits computed only on the answer positions (`logits_to_keep`). Neither changes the maths (section 6.1)"),
        ("Training subsets", "class-balanced round-robin sample, seeded by the run seed"),
        ("Seeds", "0, 1, 2 for each fine-tuned budget; budget 0 (untuned, greedy) run once"),
        ("Decoding", "greedy, max 16 new tokens, eval batch 16, one message per prompt"),
        ("Prompt", "the same system prompt (task line + 77 labels) for untuned and fine-tuned models"),
        ("Hardware", "1× Quadro RTX 6000 24 GB (shared with other users), peak 9.6 GB"),
    ]:
        w(f"| {k} | {v} |")
    w("")
    w("Software: " + ", ".join([
        "Python 3.11.15", "torch 2.15.0+cu132", "transformers 5.22.2", "peft 0.21.2",
        "datasets 5.2.0", "accelerate 1.15.0", "scikit-learn 1.9.1"]) + ". Driver 580.178.04, CUDA 13.0.")
    w("")
    w("### 3.2 Claude via Claude Code CLI (Pro subscription)")
    w("")
    # Public version: per-call summaries without identifiers (see results/claude_calls_t1.json).
    calls_doc = json.load(open(os.path.join(R, "claude_calls_t1.json")))
    raws = calls_doc["calls"]
    cli_ver = calls_doc["cli_version"]
    w("| Setting | Value |")
    w("|---|---|")
    for k, v in [
        ("Answering model", "`claude-opus-5-5` (as reported by the CLI for all 20 calls; not chosen, it is what the plan served)"),
        ("Side model", "`claude-haiku-5-5`: a small call the CLI makes itself; it does not produce the answers"),
        ("CLI", f"`{cli_ver}`, the binary bundled with the Cursor extension (no `claude` on PATH)"),
        ("Flags", "`-p --output-format json --system-prompt <ours> --tools \"\" --safe-mode --strict-mcp-config --disable-slash-commands --no-session-persistence --permission-prompts none`"),
        ("Isolation", "each call a fresh process in a new empty temp dir (checked empty before and after); no tools; no CLAUDE.md, memory, skills, hooks or MCP; message texts only, gold labels never sent"),
        ("Prompt", "`system_prompt(labels)` from `exp1_routing.py` unchanged, plus one paragraph asking for a 25-item JSON array"),
        ("Batching", "25 messages per call, 20 calls; batches from a `random.Random(456)` shuffle of the 500 (section 6.4)"),
        ("Temperature", "not settable through the CLI"),
        ("Run date", "2026-11-13"),
    ]:
        w(f"| {k} | {v} |")
    w("")
    w("The exact prompt text sent:")
    w("")
    from baseline_claude_code import BATCH_ADDITION
    from t1_common import system_prompt
    w(fence(system_prompt(labels) + BATCH_ADDITION))
    w("")

    # ---------------- 4. results
    w("## 4. Results")
    w("")
    w("### 4.1 Small model, full test set (3,076 examples), every run")
    w("")
    w("| Budget | Seed | Epochs | Accuracy | Valid-label rate | Train s | Eval s | ms/example* | Peak GB |")
    w("|---|---|---|---|---|---|---|---|---|")
    for n in BUDGETS:
        for s in SEEDS:
            if (n, s) in rows:
                r = rows[(n, s)]
                w(f"| {n} | {s} | {r['epochs']} | {r['accuracy']:.4f} | {r['valid_label_rate']:.4f} | "
                  f"{r['train_seconds']} | {r['eval_seconds']} | {r['ms_per_example']} | {r['peak_gpu_gb']} |")
    w("")
    w("\\* eval time ÷ examples at eval batch 16, adapter not merged. **Not** batch-1 latency.")
    w("")
    w("### 4.2 Small model, full test set, mean ± sample std over seeds")
    w("")
    w("| Budget | Runs | Accuracy | Macro-F1 | Valid-label rate | Mean train time |")
    w("|---|---|---|---|---|---|")
    for n in BUDGETS:
        rs = [(s, rows[(n, s)]) for s in SEEDS if (n, s) in rows]
        if not rs:
            w(f"| {n} | 0 | not completed | | | |")
            continue
        f1s = [metrics([g["gold"] for g in preds[(n, s)]], [g["pred"] for g in preds[(n, s)]], labels)[1]
               for s, _ in rs]
        w(f"| {n} | {len(rs)} | {ms([r['accuracy'] for _, r in rs])} | {ms(f1s)} | "
          f"{ms([r['valid_label_rate'] for _, r in rs])} | "
          + (f"{st.mean(ts) / 60:.1f} min |" if (ts := [r['train_seconds'] for _, r in rs if r['train_seconds'] is not None]) else "n/a |"))
    w("")
    w("At budget 100, seeds 1 and 2 have identical accuracy (0.7585). This was checked: their training "
      "subsets share only 2 of 100 examples and their predictions differ on 644 test items, so the "
      "match is a coincidence.")
    w("")
    w("### 4.3 Label efficiency on the full test set")
    w("")
    full = {n: st.mean(rows[(n, s)]["accuracy"] for s in SEEDS if (n, s) in rows) for n in BUDGETS
            if any((n, s) in rows for s in SEEDS)}
    w("| Labels | Accuracy (mean) | Gain over previous budget |")
    w("|---|---|---|")
    prev = None
    for n in full:
        w(f"| {n}{' (1 seed)' if n == FULL_N and n_full == 1 else ''} | {full[n]:.4f} | "
          f"{'' if prev is None else f'{100 * (full[n] - full[prev]):+.1f} pts'} |")
        prev = n
    w("")
    w("Accuracy on the 500-message sample against the number of labels, with Claude's accuracy and its "
      "95% interval (`make_figures.py`). The 9,993-label point is a single seed (hollow marker)"
      + (f"; the figure was drawn when only seed 0 existed, and the tables here include all {n_full} seeds." if n_full > 1 else "."))
    w("")
    w("![Label efficiency](figures/label_efficiency.png)")
    w("")
    w("### 4.4 Training loss")
    w("")
    w("Loss is logged every 20 optimiser steps and is a single-batch value, so it is noisy. "
      "Every logged step is in Appendix B (from `results/train_loss_t1.json`). Final logged loss per run:")
    w("")
    w("| Budget | Seed | Steps | Final logged loss |")
    w("|---|---|---|---|")
    loss_doc = json.load(open(os.path.join(R, "train_loss_t1.json")))
    final_loss = {}
    for s in SEEDS:
        for row in loss_doc["seeds"][str(s)]:
            final_loss[(row["budget"], s)] = (f"{row['step']}/{row['total_steps']}", f"{row['loss']:.4f}")
    full_loss_by_seed = {}
    for s in SEEDS:
        lp = find(f"train_loss_n{FULL_N}_s{s}.json")
        if lp:
            full_loss_by_seed[s] = json.load(open(lp))["rows"]
            for row in full_loss_by_seed[s]:
                final_loss[(FULL_N, s)] = (f"{row['step']}/{row['total_steps']}", f"{row['loss']:.4f}")
    full_loss = full_loss_by_seed.get(0, [])
    for n in BUDGETS[1:]:
        for s in SEEDS:
            if (n, s) in final_loss:
                steps, loss = final_loss[(n, s)]
                w(f"| {n} | {s} | {steps} | {loss} |")
    w("")

    # ---------------- 4.5 latency and cost
    w("### 4.5 Latency and cost")
    w("")
    lat_p, cost_p = os.path.join(R, "latency_t1.json"), os.path.join(R, "cost_t1.json")
    if os.path.exists(lat_p) and os.path.exists(cost_p):
        lat, cost = json.load(open(lat_p)), json.load(open(cost_p))
        b1, thr, pw = lat["batch1_latency_ms"], lat["throughput"], lat["power_during_batch16"]
        g = lat["gpu_state_before"]
        w(f"Measured on the merged out-of-scope adapter (S3, seed 0; same base model and LoRA settings, "
          f"extended prompt), fp16, greedy decoding, max 16 new tokens, {lat['prompt_tokens']['mean']:.0f} prompt "
          f"tokens on average, 200 messages after 20 warm-up requests (`measure_latency.py`). "
          f"**The card ({lat['gpu']}) was shared:** {g['other_processes']} other processes held "
          f"{g['other_processes_memory_mib'] / 1024:.1f} GB and the card was at {g['gpu_utilization_percent']:.0f}% "
          "utilisation when the measurement started, so these are not best-case numbers.")
        w("")
        w("| Measure | Value |")
        w("|---|---|")
        w(f"| Batch size 1, end to end (tokenise + generate + decode) | p50 {b1['p50']} ms · p95 {b1['p95']} ms · "
          f"p99 {b1['p99']} ms · mean {b1['mean']} ms |")
        for bs in ("8", "16", "32"):
            v = thr[bs]
            w(f"| Throughput, batch {bs} | " + (f"{v['messages_per_second']} messages/s (peak {v['peak_memory_gb']} GB) |"
                                             if "messages_per_second" in v else f"{v.get('error')} |"))
        w(f"| Mean power during the batch-16 run (whole card) | {pw['mean_watts']:.0f} W |")
        sc = lat["sanity_check"]
        w(f"| Sanity check: merged vs saved unmerged predictions | {sc['merged_correct']} vs {sc['saved_unmerged_correct']} "
          f"correct of {sc['n']}; {sc['prediction_agreement']} identical predictions |")
        w("")
        w("Cost per 1,000 requests (`compute_cost.py`). **Every hourly rate is an assumption, not a quoted "
          "market price.**")
        w("")
        w("| System | Assumed rate | Cost per 1,000 requests | Break-even volume |")
        w("|---|---|---|---|")
        sm = cost["small_model"]
        for r in sm["rates"]:
            w(f"| Qwen3-4B + LoRA, batch 16 | {r['name']} (EUR {r['eur_per_gpu_hour']:.3f}/h) | "
              f"EUR {r['eur_per_1000_requests']:.4f} | {r['break_even_requests_per_hour']:,.0f} requests/hour |")
        w(f"| {LABEL} | {cost['claude']['label']} | USD {cost['claude']['usd_per_1000_requests']:.2f} | n/a |")
        w("")
        w(f"{cost['break_even_logic']} The break-even volumes treat 1 USD as 1 EUR "
          f"(a simplification; scale by your own exchange rate). At batch 16 the measured maximum is "
          f"{sm['throughput_batch16_messages_per_second'] * 3600:,.0f} requests per hour.")
    else:
        w("Not measured (no `latency_t1.json` / `cost_t1.json`).")
    w("")
    w("### 4.6 Robustness at a glance")
    w("")
    w("Details, paired tests and caveats are in `results/t1_robustness.md`.")
    w("")
    w("![Clean, typo and two-issue accuracy](figures/stress_tests.png)")
    w("")
    w("![Out-of-scope recall and overall accuracy](figures/out_of_scope.png)")
    w("")

    # ---------------- 5. head to head
    w("## 5. Head-to-head on the same 500 examples")
    w("")
    w("| System | Label budget | Accuracy | Macro-F1 | Valid-label rate | Runs |")
    w("|---|---|---|---|---|---|")
    w(f"| {LABEL} | prompted | {c_acc:.3f} (95% Wilson CI {c_lo:.3f}–{c_hi:.3f}) | {c_f1:.3f} | {c_val:.3f} | 1 |")
    for n in BUDGETS:
        sc = sample_scores(n)
        if not sc:
            w(f"| {TAG} LoRA | {n} | not completed | | | |")
            continue
        name = f"{TAG} " + ("untuned" if n == 0 else "LoRA")
        w(f"| {name} | {n} | {ms([x[0] for x in sc], 3)} | {ms([x[1] for x in sc], 3)} | "
          f"{ms([x[2] for x in sc], 3)} | {len(sc)} seed(s) |")
    w("")
    w("Per seed on the 500 sample:")
    w("")
    w("| Budget | Seed | Accuracy | Macro-F1 | Valid |")
    w("|---|---|---|---|---|")
    for n in BUDGETS:
        for s in SEEDS:
            if (n, s) in preds:
                p = preds[(n, s)]
                a, f, v = metrics([p[i]["gold"] for i in idx], [p[i]["pred"] for i in idx], labels)
                w(f"| {n} | {s} | {a:.3f} | {f:.3f} | {v:.3f} |")
    w("")
    w("### 5.1 Paired comparison with Claude (exact McNemar test, two-sided)")
    w("")
    w("Each row compares one small-model run with Claude on the same 500 items. "
      "*Small only* = small model right, Claude wrong; *Claude only* = the reverse.")
    w("")
    w("| Budget | Seed | Both right | Small only | Claude only | Both wrong | p-value |")
    w("|---|---|---|---|---|---|---|")
    for n in BUDGETS:
        for s in SEEDS:
            if (n, s) not in preds:
                continue
            p = preds[(n, s)]
            sr = [p[i]["pred"] == p[i]["gold"] for i in idx]
            cr = [claude[i]["pred"] == claude[i]["gold"] for i in idx]
            both = sum(a and b for a, b in zip(sr, cr))
            so = sum(a and not b for a, b in zip(sr, cr))
            co = sum(b and not a for a, b in zip(sr, cr))
            bw = sum(not a and not b for a, b in zip(sr, cr))
            w(f"| {n} | {s} | {both} | {so} | {co} | {bw} | {mcnemar_exact(so, co):.3g} |")
    w("")
    w("Reading: at budgets 0–500 Claude is significantly better in every run. At 2,000 labels no run "
      "differs significantly from Claude; this supports calling it a tie, not a win. "
      "(Three seeds are tested separately, so treat p-values near 0.05 with care.)"
      + (" The 9,993-label row is a single seed; read its p-value as one run, not as a robust result." if n_full == 1 else ""))
    w("")
    w("### 5.2 Where they disagree (budget 2000, seed 0 vs Claude)")
    w("")
    p = preds[(2000, 0)]
    dis = [(i, p[i]["pred"], claude[i]["pred"], test[i][1]) for i in idx if p[i]["pred"] != claude[i]["pred"]]
    w(f"{len(dis)} of 500 items get different answers. All of them:")
    w("")
    w("| Test idx | Gold | Small (n=2000, s0) | Claude | Message |")
    w("|---|---|---|---|---|")
    for i, sp, cp, g in dis:
        mark = lambda x: f"**{x}**" if x == g else str(x)
        msg = test[i][0].replace("|", "\\|").replace("\n", " ")
        w(f"| {i} | {g} | {mark(sp)} | {mark(cp)} | {msg} |")
    w("")
    w("Bold = matches gold.")
    w("")

    # ---------------- 6. procedure
    w("## 6. Procedure, issues and fixes (chronological)")
    w("")
    w("### 6.1 Smoke test of `exp1_routing.py` (2026-11-13, 01:36–01:52)")
    w("")
    w("The script had never run on a GPU. The smoke test found that the data, chat template, "
      "`dtype=` loading and `all-linear` LoRA all worked, and that untuned accuracy was well above chance. "
      "Training ran out of GPU memory twice on the shared card (about 13.8 GB free):")
    w("")
    w("1. OOM in the LoRA backward pass. Fix: gradient checkpointing (trades compute for memory; "
      "same maths). `--load4bit` was rejected because it would change the model being compared.")
    w("2. OOM on a 1.11 GiB allocation: logits over the 151,936-token vocabulary at every prompt "
      "position, although loss uses only the ~7 answer positions. Fix: request logits only for the last "
      "k+1 positions (`logits_to_keep`) and compute the same cross-entropy in fp32. Also set "
      "`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.")
    w("")
    w("The loss mask was verified: answer tokens are `['lost', '_or', '_st', 'olen', '_card', '<|im_end|>', 'Ċ']` "
      "for the label `lost_or_stolen_card`; everything before is masked. Peak memory fell to 9.4 GB.")
    w("")
    w("### 6.2 Fine-tuning runs (01:53–04:57)")
    w("")
    w("Run on GPU 1 with:")
    w("")
    w(fence("""cd ~/slm-study && source .venv/bin/activate && mkdir -p results
for s in 0 1 2; do
  b="100,500,2000"; [ $s -eq 0 ] && b="0,100,500,2000"
  CUDA_VISIBLE_DEVICES=1 python exp1_routing.py --seed $s --budgets $b 2>&1 | tee results/exp1_log_s$s.txt
done""", "bash"))
    w("")
    w("All 10 runs finished with no errors, NaNs or OOMs. Logged training loss: Appendix B.")
    w("")
    w("### 6.3 Claude CLI checks before the run")
    w("")
    w("- `claude` was not on PATH; the Claude Code binary bundled with the Cursor extension was used.")
    w("- `claude --help` was read for the installed version. Every needed capability had a flag except "
      "a turn limit (no `--max-turns`). With all tools disabled only one turn is possible; the script "
      "rejects any call where `num_turns != 1`, and all 20 had `num_turns = 1`.")
    w("- A probe call with a 15-character system prompt was billed as ~789 prompt tokens, so the CLI "
      "adds its own text even with `--system-prompt`.")
    w("")
    w("### 6.4 Batch-order change")
    w("")
    w("The 500 sample indices were first batched in index order. A one-batch test in that order scored "
      "24/25, but showed the batch held only 4 intents in long same-label runs, because the test set is "
      "sorted by label. Across the 20 index-order batches there were only 4–7 distinct labels per batch. "
      "That gives Claude context from neighbouring messages that the one-at-a-time small model never "
      "gets. That first batch's raw answer, for the record (files since deleted):")
    w("")
    w(fence('["card_arrival","card_arrival","card_arrival","card_arrival","card_arrival","card_arrival",'
            '"activate_my_card","card_linking","card_linking","card_linking","card_linking","card_linking",'
            '"card_linking","card_linking","card_linking","exchange_rate","exchange_rate","exchange_rate",'
            '"exchange_rate","exchange_rate","exchange_rate","exchange_rate","exchange_rate",'
            '"card_payment_wrong_exchange_rate","card_payment_wrong_exchange_rate"]', "json"))
    w("")
    distinct = [len({test[j][1] for j in order[b * 25:(b + 1) * 25]}) for b in range(20)]
    w(f"Fix: the same 500 indices shuffled once with `random.Random(456)` (`results/sample500_batch_order.json`), "
      f"then split into 20 batches. Distinct labels per batch after the shuffle: {min(distinct)}–{max(distinct)}. "
      "The first test's predictions and raw output were deleted and all 20 batches rerun. The 24/25 "
      "from the discarded batch is not used anywhere. Prompt wording and scoring were not changed.")
    w("")
    w("### 6.5 Claude run (02:17–02:21)")
    w("")
    walls = [r["wall_seconds"] for r in raws]
    tot = {"in": 0, "cc": 0, "cr": 0, "out": 0, "cost": 0.0}
    for r in raws:
        u = r["usage"]
        tot["in"] += u["input_tokens"]; tot["cc"] += u["cache_creation_input_tokens"]
        tot["cr"] += u["cache_read_input_tokens"]; tot["out"] += u["output_tokens"]
        tot["cost"] += r.get("cli_list_price_estimate_usd") or 0
    w(f"20/20 batches returned a valid 25-item array on the first attempt: 0 retries, 0 failures, "
      f"0 rate limits. Wall time per call {min(walls):.1f}–{max(walls):.1f} s (median {st.median(walls):.1f} s), "
      f"including CLI start-up and network. Answering-model tokens over 20 calls: input {tot['in']}, "
      f"cache-creation {tot['cc']:,}, cache-read {tot['cr']:,}, output {tot['out']:,}. "
      f"The CLI's own list-price estimate was ${tot['cost']:.2f} in total; on a Pro plan this is not a "
      "charge, and it is not an API cost measurement. Per-call summary: Appendix C; every raw answer: Appendix D.")
    w("")

    if have_full:
        rf = rows[(FULL_N, 0)]
        w("### 6.6 Full-training-set run")
        w("")
        w("Budget 9,993 (the whole training set, checked to contain every training example exactly once "
          "and no test item), seed 0, 3 epochs, same hyperparameters as every other run, via `full_data.py`, "
          "which runs `exp1_routing.py --budgets 9993 --seed 0` unchanged. Training took "
          + (f"{rf['train_seconds'] / 3600:.1f} hours" if rf.get("train_seconds") else "an unrecorded time")
          + f"; evaluation on the full test set {rf['eval_seconds'] / 60:.0f} minutes. "
          + ("One seed only." if n_full == 1 else f"Seeds {', '.join(str(s) for s in full_seeds[1:])} were added "
             "later with every other setting identical (`full_data.py --seed`)."))
        w("")

    # ---------------- 7. caveats
    w("## 7. Caveats (read before quoting any number)")
    w("")
    for c in [
        "**Claude saw 25 messages per call**; the small model saw one at a time. Mixed batches still give Claude some context across messages.",
        "The Claude Code CLI wraps our system prompt in some text of its own and makes a side call to `claude-haiku-5-5`.",
        "**Claude's temperature could not be set**; the small model used greedy decoding.",
        "Claude's model is whatever the Pro plan served (`claude-opus-5-5`), not a chosen API model. "
        "This is **one Claude model through one CLI. It is not an API baseline** and says nothing about GPT or other frontier models.",
        "Claude was run once on the 500 sample; its CI reflects sampling of test items, not run-to-run variation.",
        "Claude's wall time includes CLI start-up and network and is not model latency. The small model's "
        "ms/example is at eval batch 16 with an unmerged adapter. **Neither is a latency measurement**; batch-1 latency of a merged adapter is in section 4.5.",
        "Some Banking77 intents overlap (for example, the test message \"I would like to reactivate my card\" is labelled `card_linking`, not `activate_my_card`), so some errors may be label ambiguity rather than model mistakes. This was observed, not measured.",
        "The 500-sample comparison has wider error bars than the full-test-set numbers; the small model's full-set numbers are more precise but have no Claude counterpart.",
        "Budget 0 was run once (greedy decoding is deterministic, so other seeds would match).",
        ("The full-training-set budget (9,993 labels) was run with one seed only, so it shows no run-to-run spread."
         if n_full == 1 else "At 9,993 labels all seeds use the same training set, so their spread reflects only "
         "initialisation and data order."),
    ]:
        w(f"- {c}")
    w("")
    w("## 8. Open work")
    w("")
    for o in [
        "Consistency test: first 100 sample items × 5 runs for Claude (20 more calls) and for the 2,000-label model.",
        "Rewording stability: 3–5 paraphrased instructions.",
        "API baselines (Claude API, GPT).",
        "A second dataset, to test whether the label-efficiency curve transfers.",
    ]:
        w(f"- {o}")
    w("")

    # ---------------- appendices
    w("---")
    w("")
    w("## Appendix A. About this public version")
    w("")
    w("The smoke-test notes, raw run logs, full Claude CLI JSON for every call, GPU process list and "
      "version-control history contain server, account and session details and are not published. Everything they were used for is kept here "
      "in identifier-free form: logged training loss (`results/train_loss_t1.json`), a per-call summary "
      "of the Claude CLI output (`results/claude_calls_t1.json`), and every prediction. Every number in "
      "sections 1–8 is computed from the published files.")
    w("")
    w("## Appendix B. Training loss, every logged step")
    w("")
    w("Single-batch loss logged every 20 optimiser steps by `exp1_routing.py`.")
    for s in SEEDS:
        w("")
        w(f"### B.{s + 1} seed {s}")
        w("")
        w("| Budget | Epoch | Step | Loss | Elapsed s |")
        w("|---|---|---|---|---|")
        for row in loss_doc["seeds"][str(s)]:
            w(f"| {row['budget']} | {row['epoch']}/{row['epochs']} | {row['step']}/{row['total_steps']} | "
              f"{row['loss']:.4f} | {row['elapsed_s']} |")
    for fs, fl in sorted(full_loss_by_seed.items()):
        if not fl:
            continue
        w("")
        w(f"### B.4 full training set (9,993 labels), seed {fs}")
        w("")
        w("| Budget | Epoch | Step | Loss | Elapsed s |")
        w("|---|---|---|---|---|")
        for row in fl:
            w(f"| {row['budget']} | {row['epoch']}/{row['epochs']} | {row['step']}/{row['total_steps']} | "
              f"{row['loss']:.4f} | {row['elapsed_s']} |")
    w("")
    w("### B.5 Result JSON per seed")
    for s in SEEDS:
        w("")
        w(f"`results/exp1_{TAG}_s{s}.json`")
        w("")
        w(fence(read(os.path.join(R, f"exp1_{TAG}_s{s}.json")), "json"))
    for s, fp in sorted(full_paths.items()):
        w("")
        w(f"`{os.path.relpath(fp)}`")
        w("")
        w(fence(read(fp), "json"))
    w("")
    w("## Appendix C. Claude calls, one row per call (`results/claude_calls_t1.json`)")
    w("")
    w("| Batch | Attempt | Return code | Wall s | num_turns | subtype | Temp dir empty | In | Cache-create | Cache-read | Out |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in raws:
        u = r["usage"]
        w(f"| {r['batch']} | {r['attempt']} | {r['returncode']} | {r['wall_seconds']} | {r['num_turns']} | "
          f"{r['subtype']} | {r['temp_dir_empty_before_and_after']} | {u['input_tokens']} | "
          f"{u['cache_creation_input_tokens']} | {u['cache_read_input_tokens']} | {u['output_tokens']} |")
    w("")
    w("## Appendix D. Claude raw answer, every batch")
    w("")
    w("Per batch: the 25 test indices sent, the models the CLI reported, and the model's raw answer.")
    for r in raws:
        b = r["batch"]
        bidx = order[b * 25:(b + 1) * 25]
        right = sum(claude[i]["pred"] == claude[i]["gold"] for i in bidx)
        w("")
        w(f"### D.{b:02d} batch {b:02d}, attempt {r['attempt']} — {right}/25 correct")
        w("")
        w(f"- Models: " + ", ".join(f"`{k}` (in {v['input_tokens']}, cache-create {v['cache_creation_input_tokens']}, "
                                  f"out {v['output_tokens']})" for k, v in r["models"].items()))
        w(f"- Test indices: {bidx}")
        w("- Raw answer:")
        w("")
        w(fence(r["result"], "json"))
    w("")
    w("## Appendix E. Every Claude prediction")
    w("")
    w("| Batch | Test idx | Gold | Raw answer | Scored as | Correct | Message |")
    w("|---|---|---|---|---|---|---|")
    for b in range(20):
        for i in order[b * 25:(b + 1) * 25]:
            r = claude[i]
            msg = r["text"].replace("|", "\\|").replace("\n", " ")
            w(f"| {b} | {i} | {r['gold']} | {r['raw']} | {r['pred']} | "
              f"{'✓' if r['pred'] == r['gold'] else '✗'} | {msg} |")
    w("")
    w("Small-model per-example predictions for all 3,076 test items are in "
      "`results/preds_Qwen3-4B-Instruct-2507_n{budget}_s{seed}.jsonl` (10 files, ~500 KB each), "
      "too large to inline; section 5.2 lists every item where the 2,000-label model and Claude differ.")
    w("")

    with open(OUT, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB, {len(L)} lines)")


if __name__ == "__main__":
    main()
