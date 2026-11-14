#!/usr/bin/env python3
"""E4b: fine-tuned encoder (answerdotai/ModernBERT-base) with a classification head.

  python next_e4b.py [--dir results/next] [--budgets 100,500,2000,9993] [--seeds 0,1,2] [--test]

Fixed settings: AdamW (PyTorch defaults otherwise), learning rate 5e-5, batch 32, max length 64,
6% warmup then linear decay, fp16 autocast with gradient scaling (a run that hits a NaN/inf loss is
repeated once in fp32 and the change recorded). Epochs: 20 for 100 labels, 10 for 500, 5 for 2,000,
3 for 9,993. No early stopping, no checkpoint saved: each model is scored and discarded. Latency is
measured for the 2,000-label seed-0 model (fixed in advance) before it is discarded.
Exit codes: 0 ok, 3 CUDA out of memory.
"""
import argparse
import gc
import json
import math
import os
import random
import statistics as st
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from t1_common import load_data, pick_subset

SYSTEM = "ModernBERT-base-ft"
MODEL = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
EPOCHS = {100: 20, 500: 10, 2000: 5, 9993: 3}
LR, BS, MAXLEN, WARMUP = 5e-5, 32, 64, 0.06


def batches(tok, texts, bs):
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        yield {k: v.cuda() for k, v in enc.items()}


def train_one(tok, sub, lab2id, seed, epochs, fp16, max_steps=0):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, num_labels=len(lab2id), id2label={v: k for k, v in lab2id.items()},
        label2id=lab2id).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps_per_epoch = math.ceil(len(sub) / BS)
    total = steps_per_epoch * epochs if not max_steps else max_steps
    warm = max(1, int(WARMUP * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    rng = random.Random(seed)
    model.train()
    step, t0, losses = 0, time.time(), []
    for ep in range(epochs):
        order = sub[:]
        rng.shuffle(order)
        for i in range(0, len(order), BS):
            chunk = order[i:i + BS]
            enc = tok([t for t, _ in chunk], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            y = torch.tensor([lab2id[l] for _, l in chunk]).cuda()
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = model(**enc, labels=y).loss
            if not torch.isfinite(loss):
                return None, None, {"nan_at_step": step}
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 50 == 0 or step == total:
                losses.append({"epoch": ep + 1, "step": step, "total_steps": total, "loss": round(loss.item(), 4)})
                print(f"    epoch {ep + 1}/{epochs} step {step}/{total} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    return model, time.time() - t0, {"losses": losses, "steps": step}


@torch.no_grad()
def predict(model, tok, texts, id2lab, bs=64):
    model.eval()
    out = []
    for enc in batches(tok, texts, bs):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits
        out += [id2lab[i] for i in logits.argmax(-1).tolist()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/next")
    ap.add_argument("--budgets", default="100,500,2000,9993")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--test", action="store_true", help="test run: 20 steps, 300 test items, latency on 40 items")
    a = ap.parse_args()
    d = a.dir
    os.makedirs(d, exist_ok=True)
    train, test, labels = load_data("mteb/banking77")
    if a.test:
        test = test[:300]
    lab2id = {l: i for i, l in enumerate(labels)}
    id2lab = {i: l for l, i in lab2id.items()}
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    rpath = os.path.join(d, f"result_{SYSTEM}.json")
    rows = json.load(open(rpath)) if os.path.exists(rpath) else []
    try:
        for n in [int(x) for x in a.budgets.split(",")]:
            for s in [int(x) for x in a.seeds.split(",")]:
                out = os.path.join(d, f"preds_{SYSTEM}_n{n}_s{s}.jsonl")
                need_latency = n == 2000 and s == 0 and not os.path.exists(os.path.join(d, "latency_e4b.json"))
                if os.path.exists(out) and any((r["budget"], r["seed"]) == (n, s) for r in rows) and not need_latency:
                    print(f"n={n} s={s} already done", flush=True)
                    continue
                sub = pick_subset(train, n, s)
                print(f"== {SYSTEM} n={n} seed={s} epochs={EPOCHS[n]}", flush=True)
                precision, note = "fp16", None
                model, secs, info = train_one(tok, sub, lab2id, s, EPOCHS[n], True, 20 if a.test else 0)
                if model is None:
                    note = f"fp16 loss became NaN/inf at step {info['nan_at_step']}; repeated in fp32"
                    print(note, flush=True)
                    precision = "fp32"
                    model, secs, info = train_one(tok, sub, lab2id, s, EPOCHS[n], False, 20 if a.test else 0)
                    if model is None:
                        raise RuntimeError(f"NaN/inf loss in fp32 as well (n={n}, s={s})")
                t0 = time.time()
                pred = predict(model, tok, [t for t, _ in test], id2lab)
                with open(out + ".tmp", "w") as f:
                    for (text, gold), p in zip(test, pred):
                        f.write(json.dumps({"text": text, "gold": gold, "raw": p, "pred": p}) + "\n")
                os.replace(out + ".tmp", out)
                acc = sum(p == g for (_, g), p in zip(test, pred)) / len(test)
                row = {"model": SYSTEM, "budget": n, "seed": s, "epochs": EPOCHS[n], "precision": precision,
                       "accuracy": round(acc, 4), "valid_label_rate": 1.0, "train_seconds": round(secs),
                       "eval_seconds": round(time.time() - t0), "steps": info["steps"],
                       "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), "test_n": len(test),
                       "logged_loss": info["losses"]}
                if note:
                    row["note"] = note
                rows = [r for r in rows if (r["budget"], r["seed"]) != (n, s)] + [row]
                json.dump(sorted(rows, key=lambda r: (r["budget"], r["seed"])), open(rpath, "w"), indent=1)
                print("RESULT", json.dumps({k: v for k, v in row.items() if k != "logged_loss"}), flush=True)
                if n == 2000 and s == 0:
                    measure_latency(model, tok, id2lab, d, 40 if a.test else 200)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    except torch.OutOfMemoryError as e:
        print(f"CUDA OOM: {e}", flush=True)
        sys.exit(3)


@torch.no_grad()
def measure_latency(model, tok, id2lab, d, n_meas):
    """Same protocol as measure_latency.py: oos_mixed items 1-20 warm-up, the next 200 measured."""
    from measure_latency import others_on_gpu, pct
    from stress_common import read_jsonl
    model.eval()
    items = read_jsonl("results/stress/oos_mixed.jsonl")[:20 + n_meas]
    warm, meas = items[:20], items[20:]
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    before = others_on_gpu(phys)
    for r in warm:
        predict(model, tok, [r["text"]], id2lab, 1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for r in meas:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict(model, tok, [r["text"]], id2lab, 1)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    predict(model, tok, [r["text"] for r in meas], id2lab, 16)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    res = {"model": f"{SYSTEM} (2,000 labels, seed 0), fp16 inference", "device": torch.cuda.get_device_name(0),
           "shared_gpu": True, "shared_gpu_note": "The card was shared with other users' jobs; these are not best-case numbers.",
           "gpu_state_before": before,
           "batch1_latency_ms": {"n": len(lat), "p50": round(pct(lat, .5), 2), "p95": round(pct(lat, .95), 2),
                                 "p99": round(pct(lat, .99), 2), "mean": round(st.mean(lat), 2)},
           "batch1_timing_scope": "tokenise + forward + argmax, torch.cuda.synchronize() around the region",
           "throughput": {"16": {"messages_per_second": round(len(meas) / secs, 1)}},
           "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}
    json.dump(res, open(os.path.join(d, "latency_e4b.json"), "w"), indent=1)
    print("LATENCY", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
