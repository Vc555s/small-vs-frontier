"""Dependency-light helpers shared by training, evaluation and the report scripts.

Moved verbatim from exp1_routing.py (which re-exports them) so that the report scripts run
without PyTorch, PEFT or Transformers installed. Behaviour is unchanged.
"""
import random
import re
from collections import defaultdict

from datasets import load_dataset


def load_data(name):
    ds = load_dataset(name)
    tr, te = ds["train"], ds["test"]
    if "label_text" in tr.column_names:
        id2name = {}
        for r in tr:
            id2name[r["label"]] = r["label_text"]
        for r in te:
            id2name.setdefault(r["label"], r["label_text"])
    else:
        id2name = dict(enumerate(tr.features["label"].names))
    train = [(r["text"], id2name[r["label"]]) for r in tr]
    test = [(r["text"], id2name[r["label"]]) for r in te]
    labels = sorted(set(id2name.values()))
    return train, test, labels

def pick_subset(train, n, seed):
    """Class-balanced sample: round-robin over classes until n examples."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for ex in train:
        by[ex[1]].append(ex)
    classes = sorted(by)
    for c in classes:
        rng.shuffle(by[c])
    rng.shuffle(classes)
    out, i = [], 0
    while len(out) < n:
        added = False
        for c in classes:
            if i < len(by[c]) and len(out) < n:
                out.append(by[c][i])
                added = True
        if not added:
            break
        i += 1
    rng.shuffle(out)
    return out

def system_prompt(labels):
    return ("You route customer messages for a bank's support desk. "
            "Reply with exactly one label from the list below and nothing else.\n\n"
            "Labels:\n" + "\n".join(labels))

def chat(tok, messages, gen):
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=gen, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)

def prompt_text(tok, sys_p, text):
    return chat(tok, [{"role": "system", "content": sys_p},
                      {"role": "user", "content": text}], gen=True)

def full_text(tok, sys_p, text, label):
    p = prompt_text(tok, sys_p, text)
    f = chat(tok, [{"role": "system", "content": sys_p},
                   {"role": "user", "content": text},
                   {"role": "assistant", "content": label}], gen=False)
    if not f.startswith(p):  # template wraps the answer in something unexpected
        f = p + label + (tok.eos_token or "")
    return p, f

def norm(s):
    s = s.strip().splitlines()[0] if s.strip() else ""
    s = s.strip().strip("\"'`*.:,;").strip().lower()
    return re.sub(r"[\s\-]+", "_", s)
