#!/usr/bin/env python3
"""E4a: frozen sentence embeddings (BAAI/bge-small-en-v1.5) + logistic regression.

  python next_e4a.py [--dir results/next] [--budgets 100,500,2000,9993] [--seeds 0,1,2] [--test]

Same training subsets as the Qwen runs: pick_subset(train, n, seed). LogisticRegression(max_iter=2000,
C=1.0), no tuning. Writes preds_<system>_n<n>_s<seed>.jsonl (full test set, test order) and
result_<system>.json; measures latency for the 2,000-label seed-0 model (fixed in advance).
"""
import argparse
import json
import os
import statistics as st
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from t1_common import load_data, pick_subset

SYSTEM = "bge-small-en-v1.5-logreg"
EMB_MODEL = "BAAI/bge-small-en-v1.5"
EMB_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def encoder(device):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL, revision=EMB_REVISION, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/next")
    ap.add_argument("--budgets", default="100,500,2000,9993")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--test", action="store_true", help="test run: 300 test items, no latency")
    a = ap.parse_args()
    d = a.dir
    os.makedirs(os.path.join(d, "emb"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test, labels = load_data("mteb/banking77")
    if a.test:
        test = test[:300]
    pos = {ex: i for i, ex in enumerate(train)}
    assert len(pos) == len(train), "duplicate training examples"

    enc = encoder(device)
    cache_tr = os.path.join(d, "emb", "train_all.npy")
    cache_te = os.path.join(d, "emb", f"test_all{'_test' if a.test else ''}.npy")
    if os.path.exists(cache_tr):
        e_tr = np.load(cache_tr)
    else:
        e_tr = enc.encode([t for t, _ in train], batch_size=128, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
        np.save(cache_tr, e_tr)
    if os.path.exists(cache_te):
        e_te = np.load(cache_te)
    else:
        e_te = enc.encode([t for t, _ in test], batch_size=128, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
        np.save(cache_te, e_te)

    rpath = os.path.join(d, f"result_{SYSTEM}.json")
    rows = json.load(open(rpath)) if os.path.exists(rpath) else []
    done = {(r["budget"], r["seed"]) for r in rows}
    keep_model = None
    for n in [int(x) for x in a.budgets.split(",")]:
        for s in [int(x) for x in a.seeds.split(",")]:
            out = os.path.join(d, f"preds_{SYSTEM}_n{n}_s{s}.jsonl")
            need_latency = n == 2000 and s == 0 and not a.test and not os.path.exists(os.path.join(d, "latency_e4a.json"))
            if (n, s) in done and os.path.exists(out) and not need_latency:
                print(f"n={n} s={s} already done", flush=True)
                continue
            sub = pick_subset(train, n, s)
            idx = [pos[ex] for ex in sub]
            t0 = time.time()
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(e_tr[idx], [lab for _, lab in sub])
            secs = time.time() - t0
            pred = clf.predict(e_te)
            with open(out + ".tmp", "w") as f:
                for (text, gold), p in zip(test, pred):
                    f.write(json.dumps({"text": text, "gold": gold, "raw": str(p), "pred": str(p)}) + "\n")
            os.replace(out + ".tmp", out)
            acc = float(np.mean([p == g for (_, g), p in zip(test, pred)]))
            row = {"model": SYSTEM, "budget": n, "seed": s, "accuracy": round(acc, 4), "valid_label_rate": 1.0,
                   "train_seconds": round(secs, 1), "n_iter": int(clf.n_iter_.max()), "test_n": len(test)}
            rows = [r for r in rows if (r["budget"], r["seed"]) != (n, s)] + [row]
            json.dump(sorted(rows, key=lambda r: (r["budget"], r["seed"])), open(rpath, "w"), indent=1)
            print("RESULT", json.dumps(row), flush=True)
            if n == 2000 and s == 0:
                keep_model = clf

    if keep_model is not None and not a.test:
        measure_latency(enc, keep_model, d, device)


def measure_latency(enc, clf, d, device):
    """Same protocol as measure_latency.py: oos_mixed items 1-20 warm-up, 21-220 measured."""
    from measure_latency import others_on_gpu, pct
    from stress_common import read_jsonl
    items = read_jsonl("results/stress/oos_mixed.jsonl")[:220]
    warm, meas = items[:20], items[20:]
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    before = others_on_gpu(phys) if device == "cuda" else None

    def run(texts):
        e = enc.encode(texts, batch_size=len(texts), normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False)
        return clf.predict(e)

    for r in warm:
        run([r["text"]])
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    lat = []
    for r in meas:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run([r["text"]])
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    for i in range(0, len(meas), 16):
        run([r["text"] for r in meas[i:i + 16]])
    if device == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    res = {"model": f"{SYSTEM} (2,000 labels, seed 0)", "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "shared_gpu": True, "shared_gpu_note": "The card was shared with other users' jobs; these are not best-case numbers.",
           "gpu_state_before": before,
           "batch1_latency_ms": {"n": len(lat), "p50": round(pct(lat, .5), 2), "p95": round(pct(lat, .95), 2),
                                 "p99": round(pct(lat, .99), 2), "mean": round(st.mean(lat), 2)},
           "batch1_timing_scope": "embed + classify, torch.cuda.synchronize() around the region",
           "throughput": {"16": {"messages_per_second": round(len(meas) / secs, 1)}},
           "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if device == "cuda" else None}
    json.dump(res, open(os.path.join(d, "latency_e4a.json"), "w"), indent=1)
    print("LATENCY", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
