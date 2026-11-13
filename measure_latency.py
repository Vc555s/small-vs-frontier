#!/usr/bin/env python3
"""Latency and throughput of the merged S3 adapter (seed 0, fixed in advance).

Model: Qwen3-4B-Instruct-2507 + LoRA adapter_S3_s0 merged with merge_and_unload(), fp16,
greedy decoding, the extended prompt (77 labels + none_of_these), max 16 new tokens.
Inputs: the first 220 items of results/stress/oos_mixed.jsonl; the first 20 are warm-up.

  CUDA_VISIBLE_DEVICES=<gpu> python measure_latency.py --adapter <path to adapter_S3_s0>

Writes results/latency_t1.json and prints a table. The GPU is shared with other users'
jobs, so these are not best-case numbers; what else was on the card is recorded.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import threading
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from exp1_routing import left_pad, load_data, load_model, norm, prompt_text
from stress_common import NONE_LABEL, SMALL_TAG, extended_prompt, read_jsonl

MODEL = f"Qwen/{SMALL_TAG}"
N_WARM, N_MEASURE = 20, 200


def smi(query, gpu, kind="gpu"):
    if kind == "gpu":
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), f"--query-gpu={query}",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return [x.strip() for x in out.strip().split(",")]
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), f"--query-compute-apps={query}",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [[y.strip() for y in x.split(",")] for x in out.strip().splitlines() if x.strip()]


def others_on_gpu(gpu):
    """Numbers only: other processes' count and memory, and GPU utilisation."""
    me = os.getpid()
    apps = smi("pid,used_memory", gpu, kind="apps")
    other = [(int(p), int(m)) for p, m in apps if p.isdigit() and int(p) != me]
    util, mem_used, mem_total = smi("utilization.gpu,memory.used,memory.total", gpu)
    return {"other_processes": len(other), "other_processes_memory_mib": sum(m for _, m in other),
            "gpu_utilization_percent": float(util), "gpu_memory_used_mib": int(mem_used),
            "gpu_memory_total_mib": int(mem_total)}


class PowerSampler(threading.Thread):
    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.gpu, self.samples, self.util, self._halt = gpu, [], [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            p, u = smi("power.draw,utilization.gpu", self.gpu)
            try:
                self.samples.append(float(p))
                self.util.append(float(u))
            except ValueError:
                pass
            self._halt.wait(1.0)

    def stop(self):
        self._halt.set()
        self.join()


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


@torch.no_grad()
def generate(model, tok, sys_p, texts, eos):
    seqs = [tok(prompt_text(tok, sys_p, t), add_special_tokens=False).input_ids for t in texts]
    ids, mask = left_pad(seqs, tok.pad_token_id)
    ids, mask = ids.cuda(), mask.cuda()
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=16, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True), [len(s) for s in seqs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to the adapter_S3_s0 directory")
    ap.add_argument("--saved_preds", default="results/stress/preds_small_S3_s0.jsonl")
    ap.add_argument("--out", default="results/latency_t1.json")
    a = ap.parse_args()
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    _, _, labels = load_data("mteb/banking77")
    allowed = list(labels) + [NONE_LABEL]
    valid = {norm(l): l for l in allowed}
    sys_p = extended_prompt(labels)
    items = read_jsonl("results/stress/oos_mixed.jsonl")[:N_WARM + N_MEASURE]
    warm, meas = items[:N_WARM], items[N_WARM:]

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(load_model(MODEL, False), a.adapter)
    model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = True
    eos = None
    gpu_name = torch.cuda.get_device_name(0)
    before = others_on_gpu(phys)

    # warm-up at batch size 1
    for r in warm:
        generate(model, tok, sys_p, [r["text"]], eos)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # batch size 1, end to end (tokenise, generate, decode)
    lat, plen, preds = [], [], []
    for r in meas:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        raw, pl = generate(model, tok, sys_p, [r["text"]], eos)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        plen += pl
        preds.append(valid.get(norm(raw[0])))
    peak_bs1 = torch.cuda.max_memory_allocated() / 2 ** 30

    # sanity: merged model vs saved (unmerged, batch 16) predictions on the same items
    saved = {r["id"]: r for r in read_jsonl(a.saved_preds)}
    gold = [r["gold"] for r in meas]
    acc_merged = sum(p == g for p, g in zip(preds, gold))
    acc_saved = sum(saved[r["id"]]["pred"] == r["gold"] for r in meas)
    agree = sum(p == saved[r["id"]]["pred"] for p, r in zip(preds, meas))
    sanity_ok = abs(acc_merged - acc_saved) <= 1

    # throughput at batch sizes 8, 16, 32 over the same 200 items
    thr = {}
    power = None
    for bs in (8, 16, 32):
        try:
            torch.cuda.synchronize()
            sampler = PowerSampler(phys) if bs == 16 else None
            if sampler:
                sampler.start()
            t0 = time.perf_counter()
            for i in range(0, len(meas), bs):
                generate(model, tok, sys_p, [r["text"] for r in meas[i:i + bs]], eos)
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            if sampler:
                sampler.stop()
                power = {"samples": len(sampler.samples),
                         "mean_watts": st.mean(sampler.samples) if sampler.samples else None,
                         "max_watts": max(sampler.samples) if sampler.samples else None,
                         "mean_gpu_utilization_percent": st.mean(sampler.util) if sampler.util else None,
                         "note": "whole-GPU power draw (nvidia-smi power.draw, 1 Hz), includes any "
                                 "other process on the card"}
            thr[str(bs)] = {"seconds": round(secs, 3), "messages_per_second": round(len(meas) / secs, 2),
                            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            thr[str(bs)] = {"error": "CUDA out of memory"}
    after = others_on_gpu(phys)

    res = {
        "model": f"{MODEL} + LoRA adapter_S3_s0 merged (merge_and_unload), fp16, greedy, max 16 new tokens",
        "prompt": "extended prompt (77 labels + none_of_these)",
        "inputs": f"results/stress/oos_mixed.jsonl items 1-{N_WARM} warm-up, {N_WARM + 1}-{N_WARM + N_MEASURE} measured",
        "gpu": gpu_name, "shared_gpu": True,
        "shared_gpu_note": "The card was shared with other users' jobs; these are not best-case numbers.",
        "gpu_state_before": before, "gpu_state_after": after,
        "batch1_latency_ms": {"n": len(lat), "p50": round(pct(lat, .5), 1), "p95": round(pct(lat, .95), 1),
                              "p99": round(pct(lat, .99), 1), "mean": round(st.mean(lat), 1),
                              "min": round(min(lat), 1), "max": round(max(lat), 1)},
        "batch1_timing_scope": "tokenise + generate + decode, torch.cuda.synchronize() around the region",
        "prompt_tokens": {"mean": round(st.mean(plen), 1), "max": max(plen)},
        "batch1_peak_memory_gb": round(peak_bs1, 2),
        "throughput": thr, "power_during_batch16": power,
        "sanity_check": {"merged_correct": acc_merged, "saved_unmerged_correct": acc_saved,
                         "n": len(meas), "prediction_agreement": agree,
                         "rule": "accuracy must match within 1 item", "passed": sanity_ok},
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    if not sanity_ok:
        raise SystemExit("SANITY CHECK FAILED: merged model differs from the saved predictions")


if __name__ == "__main__":
    main()
