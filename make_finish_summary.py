#!/usr/bin/env python3
"""Write results/finish_summary.md: the full-training-set result, latency and cost, the model
release and its verification, the files added or changed, and any gaps. Numbers come from files.

  python make_finish_summary.py [--status_dir results/finish]
"""
import argparse
import json
import math
import os
import statistics as st
import subprocess

from stress_common import FINISH_STAGES, read_jsonl, stage_state

TAG = "Qwen3-4B-Instruct-2507"
SEEDS = [0, 1, 2]
FULL_N = 9993
HF_URL = "https://huggingface.co/vishal77796/qwen3-4b-banking77-router-lora"
DRY = os.environ.get("FINISH_DRY") == "1"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status_dir", default="results/finish_dryrun/status" if DRY else "results/finish")
    a = ap.parse_args()
    res = "results/finish_dryrun" if DRY else "results"
    out = os.path.join(res, "finish_summary.md")
    idx = json.load(open("results/sample500.json"))["indices"]
    claude = {r["idx"]: r for r in read_jsonl("results/preds_claude_code_sample500.jsonl")}
    c_ok = [claude[i]["pred"] == claude[i]["gold"] for i in idx]
    c_acc = sum(c_ok) / len(idx)

    def sample(n, s, d="results"):
        p = f"{d}/preds_{TAG}_n{n}_s{s}.jsonl"
        if not os.path.exists(p):
            return None
        rows = read_jsonl(p)
        return [rows[i]["pred"] == rows[i]["gold"] for i in idx]

    L = ["# Finishing T1: full-training-set run, latency and cost, model release", ""]

    # 1. full-data result
    L += ["## 1. Full training set (9,993 labels, seed 0)", ""]
    full = sample(FULL_N, 0)
    full_json = f"results/exp1_{TAG}_n{FULL_N}_s0.json"
    if full is not None and os.path.exists(full_json):
        fr = json.load(open(full_json))[0]
        s2000 = {s: sample(2000, s) for s in SEEDS}
        acc = lambda v: sum(v) / len(v)
        full_test_2000 = [json.load(open(f"results/exp1_{TAG}_s{s}.json")) for s in SEEDS]
        ft2000 = [r["accuracy"] for rs in full_test_2000 for r in rs if r["budget"] == 2000]
        L += ["| System | Labels | Seeds | Accuracy, 500 sample | Accuracy, full test set (3,076) |", "|---|---|---|---|---|",
              f"| Qwen3-4B + LoRA | 9,993 | 1 (seed 0) | {acc(full):.3f} | {fr['accuracy']:.4f} |",
              f"| Qwen3-4B + LoRA | 2,000 | seed 0 | {acc(s2000[0]):.3f} | {ft2000[0]:.4f} |",
              f"| Qwen3-4B + LoRA | 2,000 | mean ± std of 3 | {st.mean(acc(v) for v in s2000.values()):.3f} ± "
              f"{st.stdev(acc(v) for v in s2000.values()):.3f} | {st.mean(ft2000):.4f} ± {st.stdev(ft2000):.4f} |",
              f"| Claude via Claude Code CLI | prompted | 1 run | {c_acc:.3f} | n/a |", ""]
        b = sum(x and not y for x, y in zip(full, c_ok))
        c = sum(y and not x for x, y in zip(full, c_ok))
        b2 = sum(x and not y for x, y in zip(full, s2000[0]))
        c2 = sum(y and not x for x, y in zip(full, s2000[0]))
        L += [f"Paired comparison on the 500 sample (exact McNemar, two-sided): against Claude, the 9,993-label "
              f"model is right where Claude is wrong on {b} items and wrong where Claude is right on {c} "
              f"(p = {mcnemar(b, c):.3g}); against the 2,000-label seed-0 model, {b2} vs {c2} (p = {mcnemar(b2, c2):.3g}).",
              "",
              "One seed cannot show run-to-run spread; the 2,000-label seeds differ by up to "
              f"{100 * (max(acc(v) for v in s2000.values()) - min(acc(v) for v in s2000.values())):.1f} points "
              "on the same sample, so differences of that size between single runs are not evidence of a real effect.", ""]
    else:
        L += ["Not completed: no result file for the 9,993-label run.", ""]

    # 2. latency and cost
    L += ["## 2. Latency and cost", ""]
    if os.path.exists("results/latency_t1.json") and os.path.exists("results/cost_t1.json"):
        lat, cost = json.load(open("results/latency_t1.json")), json.load(open("results/cost_t1.json"))
        b1 = lat["batch1_latency_ms"]
        L += [f"Merged S3 adapter (seed 0), fp16, greedy, {lat['gpu']} **shared with other jobs** "
              f"(about {lat['gpu_state_before']['gpu_utilization_percent']:.0f}% utilisation from other processes at the start).", "",
              "| Measure | Value |", "|---|---|",
              f"| Batch 1 latency p50 / p95 / p99 | {b1['p50']} / {b1['p95']} / {b1['p99']} ms |"]
        for bs in ("8", "16", "32"):
            v = lat["throughput"][bs]
            L.append(f"| Throughput, batch {bs} | {v.get('messages_per_second', v.get('error'))} messages/s |")
        L += ["", "| System | Assumed rate | Cost per 1,000 requests | Break-even volume |", "|---|---|---|---|"]
        for r in cost["small_model"]["rates"]:
            L.append(f"| Qwen3-4B + LoRA | {r['name']} | EUR {r['eur_per_1000_requests']:.4f} | "
                     f"{r['break_even_requests_per_hour']:,.0f} requests/hour |")
        L += [f"| Claude via Claude Code CLI | {cost['claude']['label']} | USD {cost['claude']['usd_per_1000_requests']:.2f} | n/a |",
              "", "All hourly rates are assumptions, not quoted market prices. " + cost["break_even_logic"]
              + " Break-even volumes treat 1 USD as 1 EUR.", ""]
    else:
        L += ["Not measured.", ""]

    # 3. model release
    L += ["## 3. Model release", "", f"- Adapter: {HF_URL} (created private)."]
    vp = "results/model_card_verification.json"
    if os.path.exists(vp):
        v = json.load(open(vp))
        L.append(f"- Verification from the Hub (fresh cache for the adapter; the base model was read from an "
                 f"existing local cache): the model-card usage code on {v['n']} test items agreed with the saved "
                 f"predictions on {v['agreement_with_saved']} of {v['n']}; prompt identical to training: "
                 f"{v['prompt_identical_to_training']}.")
    else:
        L.append("- Verification from the Hub: not run.")
    L.append("")

    # 4. files
    L += ["## 4. Files added or changed (relative to `main`)", ""]
    changed = sh("git diff --name-status main -- . ':!results/finish_summary.md'").splitlines()
    new = sh("git ls-files --others --exclude-standard").splitlines()
    for line in changed:
        st_, path = line.split("\t", 1)
        L.append(f"- {'modified' if st_ == 'M' else 'added' if st_ == 'A' else st_}: `{path}`")
    for path in new:
        if path != "results/finish_summary.md":
            L.append(f"- added: `{path}`")
    L.append("- added: `results/finish_summary.md`")
    L.append("")

    # 5. gaps and deviations
    L += ["## 5. Failures, skips and deviations", ""]
    gaps = []
    for s, _, _ in FINISH_STAGES:
        if s in ("summary", "commit_push"):
            continue
        state, body = stage_state(a.status_dir, s)
        if state != "done":
            gaps.append(f"- Stage `{s}`: {state}" + (f" ({body.splitlines()[0]})" if body else ""))
    L += gaps or ["- No stage failed or was skipped."]
    P = {s: {r["id"]: r["pred"] for r in read_jsonl(f"results/stress/preds_small_S3_s{s}.jsonl")} for s in SEEDS}
    ids = list(P[0])
    pair = st.mean(sum(P[x][i] != P[y][i] for i in ids) / len(ids) for x, y in [(0, 1), (0, 2), (1, 2)])
    any3 = sum(len({P[s][i] for s in SEEDS}) > 1 for i in ids) / len(ids)
    L += [
        "- The 9,993-label run used one seed, not three.",
        "- Its result file is `results/exp1_Qwen3-4B-Instruct-2507_n9993_s0.json`, not the usual "
        "`exp1_Qwen3-4B-Instruct-2507_s0.json`, so that the existing seed-0 results for budgets 0–2,000 are not overwritten.",
        "- Latency was measured on a shared GPU while other processes were running; the numbers are not best-case.",
        "- The break-even volume needs one currency; 1 USD was treated as 1 EUR.",
        f"- The seed-to-seed change in individual predictions (S3 adapters, 700-message test set) was measured at "
        f"{pair * 100:.1f}% for a pair of seeds on average and {any3 * 100:.1f}% for any of three, and the model card "
        "states these measured values rather than a round figure.",
        "- The bar charts start at zero; the label-efficiency line chart starts at 0.5 (all values lie above 0.6).",
        "- The first upload attempt to the Hub was refused because the token lacked write permission; it "
        "succeeded with a write-enabled token.",
        "",
    ]
    os.makedirs(res, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
