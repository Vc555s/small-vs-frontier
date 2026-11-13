#!/usr/bin/env python3
"""Stage train_S3_s<seed>: retrain a LoRA adapter on the same 2,000 Banking77 examples as
the T1 seed plus the 200 frozen CLINC oos train examples labelled none_of_these, with the
extended prompt, then evaluate on oos_mixed.jsonl.

Same hyperparameters as T1 (exp1_routing.py): LoRA r16/alpha32/dropout .05/all-linear,
lr 2e-4, 6% warmup + linear decay, batch 4 x accum 2, 3 epochs, fp16 autocast + GradScaler,
gradient checkpointing, loss on answer tokens only. The adapter is saved before evaluation;
on re-run, training is skipped if the adapter directory is complete.
Exit codes: 0 ok, 3 CUDA OOM, 4 NaN/inf loss.
"""
import argparse
import gc
import os
import random
import shutil
import sys
from types import SimpleNamespace

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer

from exp1_routing import load_data, load_model, pick_subset, train
from stress_common import NONE_LABEL, SMALL_TAG, extended_prompt, read_jsonl, write_meta
from stress_eval_small import evaluate_items

MODEL = f"Qwen/{SMALL_TAG}"


def adapter_complete(p):
    return (os.path.isfile(os.path.join(p, "adapter_config.json"))
            and os.path.isfile(os.path.join(p, "adapter_model.safetensors")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dir", default="results/stress")
    ap.add_argument("--in_dir", default="")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--low_mem", action="store_true", help="OOM fallback: batch 2 x accum 4 (same effective 8)")
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--dry", action="store_true", help="40 examples, 1 epoch (5 steps), eval on 10 items")
    a = ap.parse_args()
    d, src = a.dir, (a.in_dir or a.dir)
    write_meta(d, a.stage, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "?"))
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    train_all, _, labels = load_data("mteb/banking77")
    allowed = list(labels) + [NONE_LABEL]
    sys_p = extended_prompt(labels)
    bank = pick_subset(train_all, 2000, a.seed)
    oos = [(r["text"], NONE_LABEL) for r in read_jsonl(os.path.join(src, "s3_train_oos.jsonl"))]
    examples = bank + oos
    epochs = 3
    if a.dry:
        examples = bank[:36] + oos[:4]
        epochs = 1
    targs = SimpleNamespace(bs=2 if a.low_mem else 4, accum=4 if a.low_mem else 2, lr=2e-4, seed=a.seed)
    if a.low_mem:
        write_meta(d, a.stage, note="OOM fallback: train batch 2 x accum 4")
    adapter = os.path.join(d, f"adapter_S3_s{a.seed}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        if adapter_complete(adapter):
            print(f"adapter {adapter} complete, skipping training", flush=True)
            model = PeftModel.from_pretrained(load_model(MODEL, False), adapter)
        else:
            print(f"training S3 seed {a.seed}: {len(bank) if not a.dry else 36} Banking77 + "
                  f"{len(oos) if not a.dry else 4} oos, {epochs} epochs, bs {targs.bs} x {targs.accum}", flush=True)
            model = load_model(MODEL, False)
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                     target_modules="all-linear", task_type="CAUSAL_LM"))
            model.print_trainable_parameters()
            try:
                secs = train(model, tok, examples, sys_p, targs, epochs)
            except RuntimeError as e:
                if "NaN" in str(e):
                    print(f"NaN/inf loss: {e}", flush=True)
                    sys.exit(4)
                raise
            write_meta(d, a.stage, train_seconds=round(secs))
            tmp = adapter + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(tmp)
            if os.path.exists(adapter):  # an incomplete earlier attempt
                shutil.rmtree(adapter)
            os.replace(tmp, adapter)
            print(f"saved {adapter}", flush=True)
        model.config.use_cache = True
        model.eval()
        evaluate_items(model, tok, read_jsonl(os.path.join(src, "oos_mixed.jsonl")), sys_p, allowed,
                       os.path.join(d, f"preds_small_S3_s{a.seed}.jsonl"), a.eval_bs // (2 if a.low_mem else 1),
                       10 if a.dry else 0)
        write_meta(d, a.stage, peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError as e:
        print(f"CUDA OOM: {e}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
