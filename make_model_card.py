#!/usr/bin/env python3
"""Generate model_card/README.md (the Hugging Face model card) from the result files.
Every number is read from results/; the usage code is model_card/usage.py verbatim.

  python make_model_card.py
"""
import json
import os
import statistics as st

from stress_common import NONE_LABEL, read_jsonl

REPO = "https://github.com/Vc555s/small-vs-frontier"
SEED = 0  # fixed in advance; not chosen by test score


def scores(rows):
    ins = [r for r in rows if r["gold"] != NONE_LABEL]
    oos = [r for r in rows if r["gold"] == NONE_LABEL]
    return {"in_scope_acc": sum(r["pred"] == r["gold"] for r in ins) / len(ins),
            "oos_recall": sum(r["pred"] == NONE_LABEL for r in oos) / len(oos),
            "overall_acc": sum(r["pred"] == r["gold"] for r in rows) / len(rows),
            "n_in": len(ins), "n_oos": len(oos)}


def main():
    runs = {s: read_jsonl(f"results/stress/preds_small_S3_s{s}.jsonl") for s in (0, 1, 2)}
    sc = {s: scores(v) for s, v in runs.items()}
    # how many individual predictions change when only the seed changes (pairwise mean)
    ids = [r["id"] for r in runs[0]]
    p = {s: {r["id"]: r["pred"] for r in v} for s, v in runs.items()}
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_flip = [sum(p[a][i] != p[b][i] for i in ids) / len(ids) for a, b in pairs]
    flip = st.mean(pair_flip)
    any_flip = sum(len({p[s][i] for s in (0, 1, 2)}) > 1 for i in ids) / len(ids)
    summ = json.load(open("results/stress/run_summary.json"))
    meta = next(x for x in summ["stages"] if x["stage"] == f"train_S3_s{SEED}")["meta"]
    lat = json.load(open("results/latency_t1.json"))
    usage = open("model_card/usage.py").read().strip()
    oos = read_jsonl("results/stress/oos_mixed.jsonl")
    n_bank = sum(r["gold"] != NONE_LABEL for r in oos)
    n_oos = len(oos) - n_bank
    n_s3 = len(read_jsonl("results/stress/s3_train_oos.jsonl"))

    def m(k):
        xs = [sc[s][k] for s in (0, 1, 2)]
        return f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}"

    b1 = lat["batch1_latency_ms"]
    thr = lat["throughput"]
    L = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
pipeline_tag: text-generation
language:
- en
datasets:
- PolyAI/banking77
- clinc/clinc_oos
tags:
- lora
- intent-classification
- customer-support
- banking77
---

# Qwen3-4B Banking77 router (LoRA)

A LoRA adapter for [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) that routes bank customer-support messages to one of the 77 Banking77 intents, or to `none_of_these` when a message fits none of them. Given one message, it answers with exactly one label.

Vishal Jadhav · MSc Computer Science, RPTU Kaiserslautern-Landau · code, data and full evaluation: [{REPO}]({REPO})

## Usage

Tested with transformers 5.22.2, peft 0.21.2 and torch 2.15.0 on one GPU in fp16. The prompt must be exactly the one below (77 sorted labels, then `none_of_these`), because that is what the adapter was trained on.

```python
{usage}
```

## Training data

- **Banking77**: 2,000 training examples, a class-balanced sample (round-robin over the 77 intents) drawn with seed {SEED}.
- **CLINC150 out-of-scope class**: {n_s3} out-of-scope training examples labelled `none_of_these`. Items containing banking keywords were removed by a fixed filter (see the repository).

Datasets and licences:

- Banking77: Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI @ ACL. Data by PolyAI, CC BY 4.0.
- CLINC150: Larson et al. (2019), *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP. CC BY 3.0.

## Training settings

| Setting | Value |
|---|---|
| Method | LoRA, rank 16, alpha 32, dropout 0.05, all linear layers (q, k, v, o, gate, up, down) |
| Optimiser | AdamW, learning rate 2e-4, 6% warmup then linear decay, gradient clipping 1.0 |
| Batch | 4 × 2 gradient accumulation (effective 8), 3 epochs |
| Precision | fp16 autocast with gradient scaling, gradient checkpointing |
| Loss | answer tokens only |
| Hardware | 1 × {lat['gpu']} (24 GB, shared) |
| Training time | {meta['train_seconds'] / 60:.0f} minutes (this seed), peak {meta['peak_gb']} GB |

## Results

Test set: {n_bank} Banking77 test messages plus {n_oos} CLINC150 out-of-scope test messages ({n_bank + n_oos} in total). No test item was used for training or for choosing any setting.

| Metric | This adapter (seed {SEED}) | 3 seeds, mean ± std |
|---|---|---|
| In-scope accuracy ({sc[SEED]['n_in']} Banking77 messages) | {sc[SEED]['in_scope_acc']:.3f} | {m('in_scope_acc')} |
| Out-of-scope recall ({sc[SEED]['n_oos']} CLINC150 messages) | {sc[SEED]['oos_recall']:.3f} | {m('oos_recall')} |
| Overall accuracy (all {sc[SEED]['n_in'] + sc[SEED]['n_oos']}) | {sc[SEED]['overall_acc']:.3f} | {m('overall_acc')} |

Results for typo-laden and two-issue messages, and comparisons with other systems, are in the repository reports ([robustness]({REPO}/blob/main/results/t1_robustness.md), [main results]({REPO}/blob/main/results/t1_report.md)). This card makes no claim about any other model.

## Latency

Merged adapter (`merge_and_unload()`), fp16, greedy decoding, {lat['prompt_tokens']['mean']:.0f} prompt tokens on average, measured on 200 messages after 20 warm-up requests on one {lat['gpu']}. **The card was shared with other users' jobs** (about {lat['gpu_state_before']['gpu_utilization_percent']:.0f}% utilisation from other processes when the measurement started), so these are not best-case numbers.

| Measure | Value |
|---|---|
| Batch size 1, p50 / p95 / p99 | {b1['p50']:.0f} / {b1['p95']:.0f} / {b1['p99']:.0f} ms (tokenise + generate + decode) |
| Throughput, batch 8 / 16 / 32 | {thr['8'].get('messages_per_second', 'OOM')} / {thr['16'].get('messages_per_second', 'OOM')} / {thr['32'].get('messages_per_second', 'OOM')} messages per second |
| Peak GPU memory, batch 1 / 16 | {lat['batch1_peak_memory_gb']} / {thr['16'].get('peak_memory_gb', 'n/a')} GB |

## Limitations

- English only.
- One dataset: trained and tested on Banking77 (plus CLINC150 out-of-scope messages) from the same source, so it has not been tested on other banks' traffic.
- The out-of-scope examples were mostly unrelated to banking. Near-miss out-of-scope messages (banking topics outside the 77 intents) are untested.
- Retraining with a different seed changes individual predictions even though accuracy stays similar. Measured on the {len(ids)}-message test set above with the three adapters trained identically except for the seed (0, 1, 2): two adapters disagree on {flip * 100:.1f}% of messages on average (the three seed pairs range from {min(pair_flip) * 100:.1f}% to {max(pair_flip) * 100:.1f}%), and at least one of the three gives a different answer from the others on {any_flip * 100:.1f}% of messages.
- On messages that raise two issues, it returns one label.
- Not evaluated for any production use.
"""
    os.makedirs("model_card", exist_ok=True)
    with open("model_card/README.md", "w") as f:
        f.write(L)
    print("wrote model_card/README.md")


if __name__ == "__main__":
    main()
