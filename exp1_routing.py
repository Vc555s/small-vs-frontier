#!/usr/bin/env python3
"""
Experiment 1: support-ticket routing (Banking77, 77 intents).

Compares one small open model
  - untuned (budget 0), and
  - LoRA fine-tuned on 100 / 500 / 2000 real labelled examples,
on the same test set, with exact-match scoring (no LLM judge).

Usage (smoke test first, about 2-5 minutes):
  CUDA_VISIBLE_DEVICES=1 python exp1_routing.py --smoke
Full run:
  CUDA_VISIBLE_DEVICES=1 python exp1_routing.py
If you hit CUDA out-of-memory, add:  --load4bit   (or lower --bs / --eval_bs)
"""
import argparse
import gc
import json
import math
import os
import random
import re
import time
from collections import defaultdict

# Reduces fragmentation OOMs on a GPU shared with other users' jobs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dataset", default="mteb/banking77")
    p.add_argument("--budgets", default="0,100,500,2000",
                   help="comma-separated label budgets; 0 = untuned model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_n", type=int, default=0, help="0 = full test set")
    p.add_argument("--bs", type=int, default=4, help="train batch size")
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation")
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=0, help="0 = auto (5 if budget<=200 else 3)")
    p.add_argument("--load4bit", action="store_true")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="disable gradient checkpointing (faster, needs more GPU memory)")
    p.add_argument("--smoke", action="store_true", help="tiny run to check everything works")
    p.add_argument("--out", default="results")
    return p.parse_args()


# ----------------------------------------------------------------- data
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


# --------------------------------------------------------------- prompts
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


def left_pad(seqs, pad_id, fill=None):
    m = max(len(s) for s in seqs)
    fill = pad_id if fill is None else fill
    ids = [[fill] * (m - len(s)) + s for s in seqs]
    mask = [[0] * (m - len(s)) + [1] * len(s) for s in seqs]
    return torch.tensor(ids), torch.tensor(mask)


def norm(s):
    s = s.strip().splitlines()[0] if s.strip() else ""
    s = s.strip().strip("\"'`*.:,;").strip().lower()
    return re.sub(r"[\s\-]+", "_", s)


# ----------------------------------------------------------------- model
def load_model(name, load4bit):
    kw = {}
    if load4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    try:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, **kw)
    if not load4bit:
        m.to("cuda")
    return m


def train(model, tok, examples, sys_p, args, epochs):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    n_batches = math.ceil(len(examples) / args.bs)
    total = max(1, math.ceil(n_batches * epochs / args.accum))
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    rng = random.Random(args.seed)
    step, t0 = 0, time.time()
    opt.zero_grad(set_to_none=True)
    for ep in range(epochs):
        order = examples[:]
        rng.shuffle(order)
        for b in range(n_batches):
            batch = order[b * args.bs:(b + 1) * args.bs]
            seqs, labs = [], []
            for text, label in batch:
                p, f = full_text(tok, sys_p, text, label)
                p_ids = tok(p, add_special_tokens=False).input_ids
                f_ids = tok(f, add_special_tokens=False).input_ids
                seqs.append(f_ids)
                labs.append([-100] * len(p_ids) + f_ids[len(p_ids):])
            ids, mask = left_pad(seqs, tok.pad_token_id)
            lab, _ = left_pad(labs, tok.pad_token_id, fill=-100)
            ids, mask, lab = ids.cuda(), mask.cuda(), lab.cuda()
            # Left padding puts every answer at the end of its row, so only the last
            # k+1 positions carry loss. Computing logits just there avoids a
            # (batch x seq x 152k-vocab) tensor; the loss value is identical.
            k = max(len(f) - sum(1 for t in l if t == -100) for f, l in zip(seqs, labs))
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=mask,
                               logits_to_keep=k + 1).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                lab[:, -k:].reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss is NaN/inf. Try a lower --lr (e.g. 1e-4).")
            scaler.scale(loss / args.accum).backward()
            if (b + 1) % args.accum == 0 or b == n_batches - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 20 == 0 or step == total:
                    print(f"    epoch {ep + 1}/{epochs} step {step}/{total} "
                          f"loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return time.time() - t0


@torch.no_grad()
def evaluate(model, tok, data, sys_p, labels, args):
    model.eval()
    valid = {norm(l): l for l in labels}
    preds, t0 = [], time.time()
    for i in range(0, len(data), args.eval_bs):
        batch = data[i:i + args.eval_bs]
        seqs = [tok(prompt_text(tok, sys_p, t), add_special_tokens=False).input_ids
                for t, _ in batch]
        ids, mask = left_pad(seqs, tok.pad_token_id)
        ids, mask = ids.cuda(), mask.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=16,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        for (text, gold), raw in zip(batch, texts):
            preds.append({"text": text, "gold": gold, "raw": raw,
                          "pred": valid.get(norm(raw))})
        if (i // args.eval_bs) % 20 == 0:
            print(f"    eval {min(i + args.eval_bs, len(data))}/{len(data)}", flush=True)
    secs = time.time() - t0
    n = len(preds)
    acc = sum(p["pred"] == p["gold"] for p in preds) / n
    valid_rate = sum(p["pred"] is not None for p in preds) / n
    return acc, valid_rate, secs, preds


# ------------------------------------------------------------------ main
def main():
    args = get_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    if args.smoke:
        budgets, args.eval_n, args.epochs = [0, 32], 40, 1
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_all, test, labels = load_data(args.dataset)
    if args.eval_n:
        test = random.Random(123).sample(test, min(args.eval_n, len(test)))
    sys_p = system_prompt(labels)
    tag = args.model.split("/")[-1] + ("-4bit" if args.load4bit else "")
    print(f"model={args.model}  train={len(train_all)}  test={len(test)}  "
          f"labels={len(labels)}  budgets={budgets}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = []
    for n in budgets:
        print(f"\n=== budget {n} labels ===", flush=True)
        model = load_model(args.model, args.load4bit)
        train_secs, epochs = 0.0, 0
        if n > 0:
            subset = pick_subset(train_all, n, args.seed)
            epochs = args.epochs or (5 if n <= 200 else 3)
            cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                             target_modules="all-linear", task_type="CAUSAL_LM")
            if not args.no_grad_ckpt:
                # Trades compute for memory; does not change the maths of training.
                model.config.use_cache = False
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            model = get_peft_model(model, cfg)
            model.print_trainable_parameters()
            train_secs = train(model, tok, subset, sys_p, args, epochs)
            if not args.smoke:
                model.save_pretrained(os.path.join(args.out, f"adapter_{tag}_n{n}_s{args.seed}"))
        model.config.use_cache = True
        acc, valid_rate, eval_secs, preds = evaluate(model, tok, test, sys_p, labels, args)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        row = {"model": tag, "budget": n, "seed": args.seed, "epochs": epochs,
               "accuracy": round(acc, 4), "valid_label_rate": round(valid_rate, 4),
               "train_seconds": round(train_secs), "eval_seconds": round(eval_secs),
               "ms_per_example": round(1000 * eval_secs / len(test), 1),
               "peak_gpu_gb": round(peak, 1), "test_n": len(test)}
        rows.append(row)
        print("RESULT", json.dumps(row), flush=True)
        if not args.smoke:
            with open(os.path.join(args.out, f"preds_{tag}_n{n}_s{args.seed}.jsonl"), "w") as f:
                for p in preds:
                    f.write(json.dumps(p) + "\n")
            with open(os.path.join(args.out, f"exp1_{tag}_s{args.seed}.json"), "w") as f:
                json.dump(rows, f, indent=2)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("\nbudget  accuracy  valid  train_s  eval_s  peak_GB")
    for r in rows:
        print(f"{r['budget']:>6}  {r['accuracy']:.4f}    {r['valid_label_rate']:.3f}  "
              f"{r['train_seconds']:>7}  {r['eval_seconds']:>6}  {r['peak_gpu_gb']:>7}")


if __name__ == "__main__":
    main()
