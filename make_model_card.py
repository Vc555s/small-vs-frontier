#!/usr/bin/env python3
"""Generate the two Hugging Face model cards from the result files:
  model_card/README.md       qwen3-4b-banking77-router-lora (2,000 labels + out-of-scope examples)
  model_card_full/README.md  qwen3-4b-banking77-full-lora   (all 9,993 training labels)
Every number is read from results/; the usage code is each folder's usage.py verbatim.

  python make_model_card.py
"""
import json
import os
import statistics as st

from stress_common import NONE_LABEL, read_jsonl

REPO = "https://github.com/Vc555s/small-vs-frontier"
SEED = 0  # fixed in advance; not chosen by test score
HF = "https://huggingface.co"
ROUTER_ID = "vishal77796/qwen3-4b-banking77-router-lora"
FULL_ID = "vishal77796/qwen3-4b-banking77-full-lora"
QWEN = "Qwen3-4B-Instruct-2507"


def scores(rows):
    ins = [r for r in rows if r["gold"] != NONE_LABEL]
    oos = [r for r in rows if r["gold"] == NONE_LABEL]
    return {"in_scope_acc": sum(r["pred"] == r["gold"] for r in ins) / len(ins),
            "oos_recall": sum(r["pred"] == NONE_LABEL for r in oos) / len(oos),
            "overall_acc": sum(r["pred"] == r["gold"] for r in rows) / len(rows),
            "n_in": len(ins), "n_oos": len(oos)}


def model_index(name, results):
    """YAML model-index block. results: [(dataset_name, dataset_type, config, [(metric_type, metric_name, value)])]"""
    y = ["model-index:", f"- name: {name}", "  results:"]
    for ds_name, ds_type, config, metrics in results:
        y += ["  - task:", "      type: text-classification", "      name: Intent classification",
              "    dataset:", f"      name: {ds_name}", f"      type: {ds_type}"]
        if config:
            y.append(f"      config: {config}")
        y += ["      split: test", "    metrics:"]
        for mt, mn, v in metrics:
            y += [f"    - type: {mt}", f"      value: {v:.4f}", f"      name: {mn}"]
    return "\n".join(y)


def seed_flips(p, ids):
    """p: {seed: {id: pred}}. Pairwise disagreement rates and share of items where any seed differs."""
    pair_flip = [sum(p[a][i] != p[b][i] for i in ids) / len(ids) for a, b in [(0, 1), (0, 2), (1, 2)]]
    any_flip = sum(len({p[s][i] for s in (0, 1, 2)}) > 1 for i in ids) / len(ids)
    return pair_flip, any_flip


def flip_sentence(n, pair_flip, any_flip, where):
    return (f"Retraining with a different seed changes individual predictions even though accuracy stays similar. "
            f"Measured on the {n:,}-message {where} with the three adapters trained identically except for the "
            f"seed (0, 1, 2): two adapters disagree on {st.mean(pair_flip) * 100:.1f}% of messages on average (the "
            f"three seed pairs range from {min(pair_flip) * 100:.1f}% to {max(pair_flip) * 100:.1f}%), and at least "
            f"one of the three gives a different answer from the others on {any_flip * 100:.1f}% of messages.")


BANKING77_CITE = ("Banking77: Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*, "
                  "NLP4ConvAI @ ACL. Data by PolyAI, CC BY 4.0.")
CLINC_CITE = ("CLINC150: Larson et al. (2019), *An Evaluation Dataset for Intent Classification and "
              "Out-of-Scope Prediction*, EMNLP-IJCNLP. CC BY 3.0.")
AUTHOR = ("Vishal Jadhav · MSc Computer Science, RPTU Kaiserslautern-Landau · code, data and full evaluation: "
          f"[{REPO}]({REPO})")


def settings_rows(gpu, time_line):
    return f"""| Setting | Value |
|---|---|
| Method | LoRA, rank 16, alpha 32, dropout 0.05, all linear layers (q, k, v, o, gate, up, down) |
| Optimiser | AdamW, learning rate 2e-4, 6% warmup then linear decay, gradient clipping 1.0 |
| Batch | 4 × 2 gradient accumulation (effective 8), 3 epochs |
| Precision | fp16 autocast with gradient scaling, gradient checkpointing |
| Loss | answer tokens only |
| Hardware | 1 × {gpu} (24 GB, shared) |
| Training time | {time_line} |"""


def banking77_source_note():
    """Compare the mteb/banking77 copy used in the study with the original PolyAI/banking77."""
    from collections import Counter

    from datasets import load_dataset
    m, p = load_dataset("mteb/banking77"), load_dataset("PolyAI/banking77")
    names = p["train"].features["label"].names
    out = {}
    for split in ("train", "test"):
        a = Counter((r["text"], r["label_text"]) for r in m[split])
        b = Counter((r["text"], names[r["label"]]) for r in p[split])
        assert not (a - b), f"mteb {split} has examples not in PolyAI/banking77"
        out[split] = (sum(b.values()) - sum(a.values()), sum(b.values()), sum(a.values()))
    (tr_drop, tr_all, tr_kept), (te_drop, te_all, te_kept) = out["train"], out["test"]
    return (f"Loaded from the `mteb/banking77` copy on the Hub. It is a cleaned subset of the original "
            f"`PolyAI/banking77` splits: it omits {tr_drop} of the {tr_all:,} training and {te_drop} of the "
            f"{te_all:,} test examples (duplicates, some differing only in surrounding whitespace, and a few "
            f"messages of two or three words), leaving {tr_kept:,} and {te_kept:,}. Every example it contains "
            f"appears in the original with the same label.")


def router_card(source_note):
    runs = {s: read_jsonl(f"results/stress/preds_small_S3_s{s}.jsonl") for s in (0, 1, 2)}
    sc = {s: scores(v) for s, v in runs.items()}
    # how many individual predictions change when only the seed changes (pairwise mean)
    ids = [r["id"] for r in runs[0]]
    p = {s: {r["id"]: r["pred"] for r in v} for s, v in runs.items()}
    pair_flip, any_flip = seed_flips(p, ids)
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

    mi = model_index(ROUTER_ID.split("/")[1], [
        (f"Banking77 test, {n_bank}-message sample (mteb/banking77 copy)", "mteb/banking77", None,
         [("accuracy", f"In-scope accuracy, seed {SEED} (published weights)", sc[SEED]["in_scope_acc"])]),
        (f"CLINC150 out-of-scope test, {n_oos} messages", "clinc/clinc_oos", "plus",
         [("recall", f"Out-of-scope recall, seed {SEED} (published weights)", sc[SEED]["oos_recall"])]),
    ])
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
{mi}
---

# Qwen3-4B Banking77 router (LoRA)

A LoRA adapter for [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) that routes bank customer-support messages to one of the 77 Banking77 intents, or to `none_of_these` when a message fits none of them. Given one message, it answers with exactly one label.

A sibling adapter trained on all 9,993 Banking77 training examples, without out-of-scope examples, is [{FULL_ID}]({HF}/{FULL_ID}). It always answers with one of the 77 intents.

{AUTHOR}

## Usage

Tested with transformers 5.22.2, peft 0.21.2 and torch 2.15.0 on one GPU in fp16. The prompt must be exactly the one below (77 sorted labels, then `none_of_these`), because that is what the adapter was trained on.

```python
{usage}
```

## Training data

- **Banking77**: 2,000 training examples, a class-balanced sample (round-robin over the 77 intents) drawn with seed {SEED}.
  {source_note}
- **CLINC150 out-of-scope class**: {n_s3} out-of-scope training examples labelled `none_of_these`. Items containing banking keywords were removed by a fixed filter (see the repository).

Datasets and licences:

- {BANKING77_CITE}
- {CLINC_CITE}

## Training settings

{settings_rows(lat['gpu'], f"{meta['train_seconds'] / 60:.0f} minutes (this seed), peak {meta['peak_gb']} GB")}

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
- {flip_sentence(len(ids), pair_flip, any_flip, "test set above")}
- On messages that raise two issues, it returns one label.
- Not evaluated for any production use.
"""
    return L


def full_card(source_note):
    from t1_common import load_data
    train, test, labels = load_data("mteb/banking77")
    exp = {}
    preds = {}
    for s in (0, 1, 2):
        d = "results" if s == 0 else "results/next"
        exp[s] = json.load(open(f"{d}/exp1_{QWEN}_n9993_s{s}.json"))[0]
        rows = read_jsonl(f"{d}/preds_{QWEN}_n9993_s{s}.jsonl")
        assert len(rows) == len(test) and all(r["text"] == t for r, (t, _) in zip(rows, test))
        preds[s] = [r["pred"] for r in rows]
        assert exp[s]["budget"] == 9993 and exp[s]["seed"] == s and exp[s]["test_n"] == len(test)
    gold = [g for _, g in test]
    sample = json.load(open("results/sample500.json"))["indices"]
    full_idx = list(range(len(test)))

    def acc(s, idx):
        return sum(preds[s][i] == gold[i] for i in idx) / len(idx)

    def valid(s, idx):
        return sum(preds[s][i] in labels for i in idx) / len(idx)

    def m(f, idx, d=3):
        xs = [f(s, idx) for s in (0, 1, 2)]
        return f"{st.mean(xs):.{d}f} ± {st.stdev(xs):.{d}f}"

    # the saved accuracy must equal the one recomputed from the predictions
    for s in (0, 1, 2):
        assert round(acc(s, full_idx), 4) == exp[s]["accuracy"], s
    from t1_common import normalised_overlap
    n_exact, hits = normalised_overlap(train, test)
    ov = {i for i, _, _ in hits}
    keep_full = [i for i in full_idx if i not in ov]
    keep_sample = [i for i in sample if i not in ov]
    same = all(f"{acc(s, a):.3f}" == f"{acc(s, b):.3f}" for s in (0, 1, 2)
               for a, b in ((full_idx, keep_full), (sample, keep_sample)))
    ov_sentence = (f"{len(hits)} test messages also occur in the training split once case and surrounding whitespace "
                   f"are ignored ({n_exact} match exactly; {len(sample) - len(keep_sample)} of them "
                   f"{'is' if len(sample) - len(keep_sample) == 1 else 'are'} in the {len(sample)}-message sample). ")
    if same:
        ov_sentence += "Removing them leaves the accuracy of every seed on both test sets unchanged to three decimals."
    else:
        ov_sentence += (f"Without them: {m(acc, keep_sample)} on the sample ({len(keep_sample)} messages) and "
                        f"{m(acc, keep_full)} on the full test set ({len(keep_full):,}).")
    pair_flip, any_flip = seed_flips({s: dict(enumerate(preds[s])) for s in (0, 1, 2)}, full_idx)
    lat = json.load(open("results/latency_t1.json"))
    usage = open("model_card_full/usage.py").read().strip()
    mins = [exp[s]["train_seconds"] / 60 for s in (0, 1, 2)]
    n_s = len(sample)
    mi = model_index(FULL_ID.split("/")[1], [
        (f"Banking77 test, all {len(test):,} messages (mteb/banking77 copy)", "mteb/banking77", None,
         [("accuracy", f"Accuracy, seed {SEED} (published weights)", acc(SEED, full_idx))]),
        (f"Banking77 test, {n_s}-message sample (mteb/banking77 copy)", "mteb/banking77", None,
         [("accuracy", f"Accuracy, seed {SEED} (published weights)", acc(SEED, sample))]),
    ])
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
pipeline_tag: text-generation
language:
- en
datasets:
- PolyAI/banking77
tags:
- lora
- intent-classification
- customer-support
- banking77
{mi}
---

# Qwen3-4B Banking77 intent classifier, full training set (LoRA)

A LoRA adapter for [Qwen/Qwen3-4B-Instruct-2507]({HF}/Qwen/Qwen3-4B-Instruct-2507) that assigns a bank customer-support message to one of the 77 Banking77 intents. Given one message, it answers with exactly one label. It was trained on all {len(train):,} Banking77 training examples.

**It was not trained to reject out-of-scope messages.** It has no `none_of_these` answer and will give one of the 77 intents for any input, including messages that have nothing to do with banking. For traffic that may contain such messages, use the sibling adapter [{ROUTER_ID}]({HF}/{ROUTER_ID}), which was trained with out-of-scope examples and can answer `none_of_these`.

{AUTHOR}

## Usage

Tested with transformers 5.22.2, peft 0.21.2 and torch 2.15.0 on one GPU in fp16. The prompt must be exactly the one below (the 77 sorted labels), because that is what the adapter was trained on.

```python
{usage}
```

## Training data

- **Banking77**: all {len(train):,} training examples, 3 epochs.
  {source_note}

Dataset and licence:

- {BANKING77_CITE}

## Training settings

{settings_rows(lat['gpu'], (f"{mins[SEED]:.0f} minutes (this seed; {min(mins):.0f} to {max(mins):.0f} minutes across the three seeds)"
                                  if round(min(mins)) != round(max(mins)) else
                                  f"{mins[SEED]:.0f} minutes (each of the three seeds)") + f", peak {exp[SEED]['peak_gpu_gb']} GB")}

## Results

Three adapters were trained with identical settings and seeds 0, 1 and 2. **The published weights are seed {SEED}, which was fixed before any result was seen**; it was not chosen by test score. Test data: the Banking77 test split ({len(test):,} messages) and the fixed {n_s}-message sample of it used throughout the study. No test item was used for training or for choosing any setting.

| Test set | 3 seeds, mean ± std | This adapter (seed {SEED}) |
|---|---|---|
| Accuracy, {n_s}-message sample | {m(acc, sample)} | {acc(SEED, sample):.3f} |
| Accuracy, full test set ({len(test):,}) | {m(acc, full_idx)} | {acc(SEED, full_idx):.3f} |
| Answers that are one of the 77 labels, full test set | {m(valid, full_idx, 4)} | {valid(SEED, full_idx):.4f} |

{ov_sentence} Comparisons with other systems are in the repository report ([baselines]({REPO}/blob/next-experiments/results/t1_baselines.md)). This card makes no claim about any other model.

Latency was not measured separately for this adapter. It has the same base model and LoRA size as [{ROUTER_ID}]({HF}/{ROUTER_ID}), whose card reports latency, and a slightly shorter prompt (no `none_of_these` label or instruction).

## Limitations

- **No out-of-scope answer**: every input gets one of the 77 intents (see above).
- English only.
- One dataset: trained and tested on Banking77 from the same source, so it has not been tested on other banks' traffic.
- {flip_sentence(len(test), pair_flip, any_flip, "full test set")}
- On messages that raise two issues, it returns one label.
- Not evaluated for any production use.
"""


def main():
    source_note = banking77_source_note()
    for folder, card in (("model_card", router_card(source_note)), ("model_card_full", full_card(source_note))):
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/README.md", "w") as f:
            f.write(card)
        print(f"wrote {folder}/README.md")


if __name__ == "__main__":
    main()
