#!/usr/bin/env python3
"""Build and freeze the inputs for the next experiments (before any model call).

  python next_build.py [--dir results/next]

Writes:
  clean500.jsonl               the 500 sample messages in the T1 batch order (Random(456))
  icl_one_per_class.json       E2a: one example per intent from the 2,000-example seed-0 subset
  icl_retrieved/batch_XX.json  E2b: the examples for each 25-message call (retrieved by embedding)
  emb/                         cached sentence embeddings (not committed)
All examples come from the 2,000-example seed-0 training subset; no test label is used.
"""
import argparse
import json
import os
import random

import numpy as np

from t1_common import load_data, pick_subset

EMB_MODEL = "BAAI/bge-small-en-v1.5"
EMB_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
K_NEIGHBOURS = 5
BATCH = 25


def embed(texts, device="cpu"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL, revision=EMB_REVISION, device=device)
    # Symmetric message-to-message similarity: no query instruction prefix.
    return m.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/next")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    d = a.dir
    os.makedirs(os.path.join(d, "icl_retrieved"), exist_ok=True)
    train, test, labels = load_data("mteb/banking77")
    order = json.load(open("results/sample500_batch_order.json"))["order"]

    # clean 500, same order as every earlier Claude run
    with open(os.path.join(d, "clean500.jsonl"), "w") as f:
        for i in order:
            f.write(json.dumps({"id": f"T{i}", "text": test[i][0], "gold": test[i][1], "src_idx": i}) + "\n")

    sub = pick_subset(train, 2000, 0)   # the 2,000-example seed-0 training subset

    # E2a: shuffle the subset with Random(2024); first example of each intent, in that order
    shuf = sub[:]
    random.Random(2024).shuffle(shuf)
    seen, one = set(), []
    for text, lab in shuf:
        if lab not in seen:
            seen.add(lab)
            one.append({"text": text, "label": lab})
    assert len(one) == len(labels) == 77, len(one)
    json.dump({"rule": "seed-0 2,000-example subset shuffled with random.Random(2024); first example of "
                       "each intent, kept in that shuffled order",
               "n": len(one), "examples": one}, open(os.path.join(d, "icl_one_per_class.json"), "w"), indent=1)

    # E2b: embeddings of the subset and the 500 messages; 5 nearest by cosine per message
    os.makedirs(os.path.join(d, "emb"), exist_ok=True)
    sub_emb = embed([t for t, _ in sub], a.device)
    msg_emb = embed([test[i][0] for i in order], a.device)
    np.save(os.path.join(d, "emb", "subset2000_s0.npy"), sub_emb)
    np.save(os.path.join(d, "emb", "sample500_batchorder.npy"), msg_emb)
    sims = msg_emb @ sub_emb.T
    for b in range(len(order) // BATCH):
        rows = range(b * BATCH, (b + 1) * BATCH)
        union = set()
        neigh = {}
        for r in rows:
            top = np.argsort(-sims[r], kind="stable")[:K_NEIGHBOURS].tolist()
            neigh[f"T{order[r]}"] = top
            union.update(top)
        idxs = sorted(union)                 # deterministic, then a fixed shuffle
        random.Random(7).shuffle(idxs)
        json.dump({"batch": b, "rule": f"union of the {K_NEIGHBOURS} nearest subset examples (cosine, "
                                       f"{EMB_MODEL}) of each message in the call, duplicates removed, "
                                       "sorted by subset position then shuffled with random.Random(7)",
                   "n": len(idxs), "neighbours_by_message": neigh,
                   "examples": [{"subset_pos": j, "text": sub[j][0], "label": sub[j][1]} for j in idxs]},
                  open(os.path.join(d, "icl_retrieved", f"batch_{b:02d}.json"), "w"), indent=1)
    sizes = [json.load(open(os.path.join(d, "icl_retrieved", f"batch_{b:02d}.json")))["n"] for b in range(20)]
    print(f"clean500: {len(order)} | E2a examples: {len(one)} | E2b examples per call: min {min(sizes)}, "
          f"max {max(sizes)}, mean {sum(sizes) / len(sizes):.1f}")


if __name__ == "__main__":
    main()
