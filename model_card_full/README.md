---
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
model-index:
- name: qwen3-4b-banking77-full-lora
  results:
  - task:
      type: text-classification
      name: Intent classification
    dataset:
      name: Banking77 test, all 3,076 messages (mteb/banking77 copy)
      type: mteb/banking77
      split: test
    metrics:
    - type: accuracy
      value: 0.9477
      name: Accuracy, seed 0 (published weights)
  - task:
      type: text-classification
      name: Intent classification
    dataset:
      name: Banking77 test, 500-message sample (mteb/banking77 copy)
      type: mteb/banking77
      split: test
    metrics:
    - type: accuracy
      value: 0.9380
      name: Accuracy, seed 0 (published weights)
---

# Qwen3-4B Banking77 intent classifier, full training set (LoRA)

A LoRA adapter for [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) that assigns a bank customer-support message to one of the 77 Banking77 intents. Given one message, it answers with exactly one label. It was trained on all 9,993 Banking77 training examples.

**It was not trained to reject out-of-scope messages.** It has no `none_of_these` answer and will give one of the 77 intents for any input, including messages that have nothing to do with banking. For traffic that may contain such messages, use the sibling adapter [vishal77796/qwen3-4b-banking77-router-lora](https://huggingface.co/vishal77796/qwen3-4b-banking77-router-lora), which was trained with out-of-scope examples and can answer `none_of_these`.

Vishal Jadhav · MSc Computer Science, RPTU Kaiserslautern-Landau · code, data and full evaluation: [https://github.com/Vc555s/small-vs-frontier](https://github.com/Vc555s/small-vs-frontier)

## Usage

Tested with transformers 5.22.2, peft 0.21.2 and torch 2.15.0 on one GPU in fp16. The prompt must be exactly the one below (the 77 sorted labels), because that is what the adapter was trained on.

```python
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "vishal77796/qwen3-4b-banking77-full-lora"

# The 77 Banking77 intents, sorted: the exact prompt used in training.
labels = sorted(set(load_dataset("mteb/banking77", split="train")["label_text"]))
SYSTEM = ("You route customer messages for a bank's support desk. "
          "Reply with exactly one label from the list below and nothing else.\n\n"
          "Labels:\n" + "\n".join(labels))

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(model, ADAPTER).eval()


def norm(s):
    s = s.strip().splitlines()[0] if s.strip() else ""
    s = s.strip().strip("\"'`*.:,;").strip().lower()
    return re.sub(r"[\s\-]+", "_", s)


VALID = {norm(l): l for l in labels}


@torch.no_grad()
def route(message):
    """Return one of the 77 intents (None if the output is not a label)."""
    text = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": message}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=16, do_sample=False)
    return VALID.get(norm(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)))


print(route("I still haven't received my new card, it's been two weeks."))
print(route("How do I change the PIN on my card?"))
```

## Training data

- **Banking77**: all 9,993 training examples, 3 epochs.
  Loaded from the `mteb/banking77` copy on the Hub. It is a cleaned subset of the original `PolyAI/banking77` splits: it omits 10 of the 10,003 training and 4 of the 3,080 test examples (duplicates, some differing only in surrounding whitespace, and a few messages of two or three words), leaving 9,993 and 3,076. Every example it contains appears in the original with the same label.

Dataset and licence:

- Banking77: Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI @ ACL. Data by PolyAI, CC BY 4.0.

## Training settings

| Setting | Value |
|---|---|
| Method | LoRA, rank 16, alpha 32, dropout 0.05, all linear layers (q, k, v, o, gate, up, down) |
| Optimiser | AdamW, learning rate 2e-4, 6% warmup then linear decay, gradient clipping 1.0 |
| Batch | 4 × 2 gradient accumulation (effective 8), 3 epochs |
| Precision | fp16 autocast with gradient scaling, gradient checkpointing |
| Loss | answer tokens only |
| Hardware | 1 × Quadro RTX 6000 (24 GB, shared) |
| Training time | 160 minutes (each of the three seeds), peak 9.6 GB |

## Results

Three adapters were trained with identical settings and seeds 0, 1 and 2. **The published weights are seed 0, which was fixed before any result was seen**; it was not chosen by test score. Test data: the Banking77 test split (3,076 messages) and the fixed 500-message sample of it used throughout the study. No test item was used for training or for choosing any setting.

| Test set | 3 seeds, mean ± std | This adapter (seed 0) |
|---|---|---|
| Accuracy, 500-message sample | 0.943 ± 0.005 | 0.938 |
| Accuracy, full test set (3,076) | 0.947 ± 0.003 | 0.948 |
| Answers that are one of the 77 labels, full test set | 0.9998 ± 0.0002 | 0.9997 |

6 test messages also occur in the training split once case and surrounding whitespace are ignored (0 match exactly; 1 of them is in the 500-message sample). Removing them leaves the accuracy of every seed on both test sets unchanged to three decimals. Comparisons with other systems are in the repository report ([baselines](https://github.com/Vc555s/small-vs-frontier/blob/next-experiments/results/t1_baselines.md)). This card makes no claim about any other model.

Latency was not measured separately for this adapter. It has the same base model and LoRA size as [vishal77796/qwen3-4b-banking77-router-lora](https://huggingface.co/vishal77796/qwen3-4b-banking77-router-lora), whose card reports latency, and a slightly shorter prompt (no `none_of_these` label or instruction).

## Limitations

- **No out-of-scope answer**: every input gets one of the 77 intents (see above).
- English only.
- One dataset: trained and tested on Banking77 from the same source, so it has not been tested on other banks' traffic.
- Retraining with a different seed changes individual predictions even though accuracy stays similar. Measured on the 3,076-message full test set with the three adapters trained identically except for the seed (0, 1, 2): two adapters disagree on 3.3% of messages on average (the three seed pairs range from 3.1% to 3.4%), and at least one of the three gives a different answer from the others on 4.6% of messages.
- On messages that raise two issues, it returns one label.
- Not evaluated for any production use.
