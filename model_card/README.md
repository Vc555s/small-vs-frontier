---
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
model-index:
- name: qwen3-4b-banking77-router-lora
  results:
  - task:
      type: text-classification
      name: Intent classification
    dataset:
      name: Banking77 test, 500-message sample (mteb/banking77 copy)
      type: mteb/banking77
      split: test
    metrics:
    - type: accuracy
      value: 0.9100
      name: In-scope accuracy, seed 0 (published weights)
  - task:
      type: text-classification
      name: Intent classification
    dataset:
      name: CLINC150 out-of-scope test, 200 messages
      type: clinc/clinc_oos
      config: plus
      split: test
    metrics:
    - type: recall
      value: 0.9900
      name: Out-of-scope recall, seed 0 (published weights)
---

# Qwen3-4B Banking77 router (LoRA)

A LoRA adapter for [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) that routes bank customer-support messages to one of the 77 Banking77 intents, or to `none_of_these` when a message fits none of them. Given one message, it answers with exactly one label.

A sibling adapter trained on all 9,993 Banking77 training examples, without out-of-scope examples, is [vishal77796/qwen3-4b-banking77-full-lora](https://huggingface.co/vishal77796/qwen3-4b-banking77-full-lora). It always answers with one of the 77 intents.

Vishal Jadhav · MSc Computer Science, RPTU Kaiserslautern-Landau · code, data and full evaluation: [https://github.com/Vc555s/small-vs-frontier](https://github.com/Vc555s/small-vs-frontier)

## Usage

Tested with transformers 5.22.2, peft 0.21.2 and torch 2.15.0 on one GPU in fp16. The prompt must be exactly the one below (77 sorted labels, then `none_of_these`), because that is what the adapter was trained on.

```python
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "vishal77796/qwen3-4b-banking77-router-lora"

# The 77 Banking77 intents, sorted, plus none_of_these: the exact prompt used in training.
labels = sorted(set(load_dataset("mteb/banking77", split="train")["label_text"]))
SYSTEM = ("You route customer messages for a bank's support desk. "
          "Reply with exactly one label from the list below and nothing else.\n\n"
          "Labels:\n" + "\n".join(labels + ["none_of_these"])
          + "\n\nIf a message does not fit any label, answer none_of_these.")

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(model, ADAPTER).eval()


def norm(s):
    s = s.strip().splitlines()[0] if s.strip() else ""
    s = s.strip().strip("\"'`*.:,;").strip().lower()
    return re.sub(r"[\s\-]+", "_", s)


VALID = {norm(l): l for l in labels + ["none_of_these"]}


@torch.no_grad()
def route(message):
    """Return one of the 77 intents or 'none_of_these' (None if the output is not a label)."""
    text = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": message}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=16, do_sample=False)
    return VALID.get(norm(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)))


print(route("I still haven't received my new card, it's been two weeks."))
print(route("What's the weather like in Berlin tomorrow?"))
```

## Training data

- **Banking77**: 2,000 training examples, a class-balanced sample (round-robin over the 77 intents) drawn with seed 0.
  Loaded from the `mteb/banking77` copy on the Hub. It is a cleaned subset of the original `PolyAI/banking77` splits: it omits 10 of the 10,003 training and 4 of the 3,080 test examples (duplicates, some differing only in surrounding whitespace, and a few messages of two or three words), leaving 9,993 and 3,076. Every example it contains appears in the original with the same label.
- **CLINC150 out-of-scope class**: 200 out-of-scope training examples labelled `none_of_these`. Items containing banking keywords were removed by a fixed filter (see the repository).

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
| Hardware | 1 × Quadro RTX 6000 (24 GB, shared) |
| Training time | 37 minutes (this seed), peak 10.7 GB |

## Results

Test set: 500 Banking77 test messages plus 200 CLINC150 out-of-scope test messages (700 in total). No test item was used for training or for choosing any setting.

| Metric | This adapter (seed 0) | 3 seeds, mean ± std |
|---|---|---|
| In-scope accuracy (500 Banking77 messages) | 0.910 | 0.913 ± 0.008 |
| Out-of-scope recall (200 CLINC150 messages) | 0.990 | 0.990 ± 0.005 |
| Overall accuracy (all 700) | 0.933 | 0.935 ± 0.007 |

Results for typo-laden and two-issue messages, and comparisons with other systems, are in the repository reports ([robustness](https://github.com/Vc555s/small-vs-frontier/blob/main/results/t1_robustness.md), [main results](https://github.com/Vc555s/small-vs-frontier/blob/main/results/t1_report.md)). This card makes no claim about any other model.

## Latency

Merged adapter (`merge_and_unload()`), fp16, greedy decoding, 455 prompt tokens on average, measured on 200 messages after 20 warm-up requests on one Quadro RTX 6000. **The card was shared with other users' jobs** (about 54% utilisation from other processes when the measurement started), so these are not best-case numbers.

| Measure | Value |
|---|---|
| Batch size 1, p50 / p95 / p99 | 258 / 351 / 442 ms (tokenise + generate + decode) |
| Throughput, batch 8 / 16 / 32 | 9.45 / 11.19 / 11.52 messages per second |
| Peak GPU memory, batch 1 / 16 | 7.59 / 9.12 GB |

## Limitations

- English only.
- One dataset: trained and tested on Banking77 (plus CLINC150 out-of-scope messages) from the same source, so it has not been tested on other banks' traffic.
- The out-of-scope examples were mostly unrelated to banking. Near-miss out-of-scope messages (banking topics outside the 77 intents) are untested.
- Retraining with a different seed changes individual predictions even though accuracy stays similar. Measured on the 700-message test set above with the three adapters trained identically except for the seed (0, 1, 2): two adapters disagree on 6.2% of messages on average (the three seed pairs range from 5.4% to 7.4%), and at least one of the three gives a different answer from the others on 8.9% of messages.
- On messages that raise two issues, it returns one label.
- Not evaluated for any production use.
