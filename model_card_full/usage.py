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
