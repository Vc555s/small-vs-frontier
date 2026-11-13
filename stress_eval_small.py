#!/usr/bin/env python3
"""Small model (saved 2,000-label adapters) on the stress sets.

  --mode A    typo500.jsonl,   original prompt
  --mode B    merged250.jsonl, original prompt
  --mode S1   oos_mixed.jsonl, extended prompt (none_of_these option)
  --mode S2   s2_val_s{seed}.jsonl and oos_mixed.jsonl, original prompt, with confidence;
              picks the abstention threshold on validation only, applies it once to test.

Confidence (used throughout, for S2): the probability of the greedy answer, i.e. the
product of the probabilities of its generated tokens up to and including the first
end-of-turn token (exp of the summed token log-probabilities). It is recorded for every
mode but used only by S2.

Predictions are written incrementally to <out>.partial (flushed every batch) and renamed
to <out> when complete; a re-run resumes from the last written line.
Exit codes: 0 ok, 3 CUDA out of memory.
"""
import argparse
import gc
import json
import os
import sys
import time

import torch

from exp1_routing import left_pad, load_data, load_model, norm, prompt_text, system_prompt
from stress_common import (NONE_LABEL, SEEDS, SMALL_TAG, extended_prompt, read_jsonl,
                           write_json_atomic, write_meta)

ADAPTER = "results/adapter_" + SMALL_TAG + "_n2000_s{seed}"


def load_adapter_model(path):
    from peft import PeftModel
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"Qwen/{SMALL_TAG}")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = load_model(f"Qwen/{SMALL_TAG}", False)
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    return model, tok


def stop_ids(model, tok):
    e = model.generation_config.eos_token_id
    ids = set(e if isinstance(e, list) else [e])
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


@torch.no_grad()
def evaluate_items(model, tok, items, sys_p, allowed, out, eval_bs, limit=0, log_every=10):
    """Greedy-decode every item, append one JSON line per item to out.partial, resume-safe."""
    if os.path.exists(out):
        print(f"  {out} already complete, skipping", flush=True)
        return read_jsonl(out)
    if limit:
        items = items[:limit]
    part = out + ".partial"
    done = 0
    if os.path.exists(part):
        with open(part) as f:
            lines = [l for l in f if l.strip()]
        # keep only complete JSON lines
        good = []
        for l in lines:
            try:
                good.append(json.loads(l))
            except json.JSONDecodeError:
                break
        for k, r in enumerate(good):
            assert r["id"] == items[k]["id"], f"resume mismatch at line {k}"
        done = len(good)
        with open(part, "w") as f:
            for r in good:
                f.write(json.dumps(r) + "\n")
        print(f"  resuming {out} at item {done}/{len(items)}", flush=True)
    valid = {norm(l): l for l in allowed}
    eos = stop_ids(model, tok)
    t0 = time.time()
    with open(part, "a") as f:
        for i in range(done, len(items), eval_bs):
            batch = items[i:i + eval_bs]
            seqs = [tok(prompt_text(tok, sys_p, r["text"]), add_special_tokens=False).input_ids
                    for r in batch]
            ids, mask = left_pad(seqs, tok.pad_token_id)
            ids, mask = ids.cuda(), mask.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                g = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=16,
                                   do_sample=False, pad_token_id=tok.pad_token_id,
                                   output_logits=True, return_dict_in_generate=True)
            gen = g.sequences[:, ids.shape[1]:]
            lps = torch.stack([torch.log_softmax(l.float(), dim=-1) for l in g.logits], dim=1)
            tok_lp = lps.gather(2, gen[:, :lps.shape[1]].unsqueeze(-1)).squeeze(-1)
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for b, (r, raw) in enumerate(zip(batch, texts)):
                s = 0.0
                for t in range(tok_lp.shape[1]):
                    s += tok_lp[b, t].item()
                    if gen[b, t].item() in eos:
                        break
                rec = {k: v for k, v in r.items() if k != "orig_text"}
                rec.update({"raw": raw, "pred": valid.get(norm(raw)), "logprob": s,
                            "conf": float(torch.tensor(s).exp())})
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n = min(i + eval_bs, len(items))
            if ((i - done) // eval_bs) % log_every == 0 or n == len(items):
                print(f"    {os.path.basename(out)} {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    os.replace(part, out)
    return read_jsonl(out)


def pick_threshold(val_rows):
    """Threshold t: predict none_of_these when conf < t. Maximise overall accuracy on
    validation. Candidates: 0 and every distinct confidence value. Ties -> lowest t."""
    def acc(t):
        ok = 0
        for r in val_rows:
            p = NONE_LABEL if r["conf"] < t else r["pred"]
            ok += p == r["gold"]
        return ok / len(val_rows)
    cands = sorted({0.0} | {r["conf"] for r in val_rows} | {1.0 + 1e-9})
    best = max(cands, key=lambda t: (acc(t), -t))
    return best, acc(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["A", "B", "S1", "S2"])
    ap.add_argument("--dir", default="results/stress")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--low_mem", action="store_true", help="halve eval batch size (OOM fallback)")
    ap.add_argument("--limit", type=int, default=0, help="dry run: only the first N items")
    ap.add_argument("--in_dir", default="", help="where the frozen sets are (default: --dir)")
    a = ap.parse_args()
    d = a.dir
    src = a.in_dir or d
    bs = a.eval_bs // 2 if a.low_mem else a.eval_bs
    if a.low_mem:
        write_meta(d, a.stage, note=f"OOM fallback: eval batch size {bs}")
    _, _, labels = load_data("mteb/banking77")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    write_meta(d, a.stage, gpu=gpu)
    try:
        for s in [int(x) for x in a.seeds.split(",")]:
            print(f"== {a.stage} seed {s}", flush=True)
            model, tok = load_adapter_model(ADAPTER.format(seed=s))
            if a.mode in ("A", "B"):
                fn = "typo500.jsonl" if a.mode == "A" else "merged250.jsonl"
                evaluate_items(model, tok, read_jsonl(os.path.join(src, fn)), system_prompt(labels),
                               labels, os.path.join(d, f"preds_small_{a.mode}_s{s}.jsonl"), bs, a.limit)
            elif a.mode == "S1":
                evaluate_items(model, tok, read_jsonl(os.path.join(src, "oos_mixed.jsonl")),
                               extended_prompt(labels), list(labels) + [NONE_LABEL],
                               os.path.join(d, f"preds_small_S1_s{s}.jsonl"), bs, a.limit)
            else:
                val = evaluate_items(model, tok, read_jsonl(os.path.join(src, f"s2_val_s{s}.jsonl")),
                                     system_prompt(labels), labels,
                                     os.path.join(d, f"preds_small_S2val_s{s}.jsonl"), bs,
                                     a.limit and a.limit * 2)
                tst = evaluate_items(model, tok, read_jsonl(os.path.join(src, "oos_mixed.jsonl")),
                                     system_prompt(labels), labels,
                                     os.path.join(d, f"preds_small_S2raw_s{s}.jsonl"), bs, a.limit)
                tp = os.path.join(d, "s2_thresholds.json")
                th = json.load(open(tp)) if os.path.exists(tp) else {}
                if str(s) not in th:   # frozen once chosen
                    t, vacc = pick_threshold(val)
                    th[str(s)] = {"threshold": t, "val_accuracy": vacc, "val_n": len(val),
                                  "rule": "none_of_these if conf < threshold; conf = P(greedy answer)"}
                    write_json_atomic(tp, th)
                t = th[str(s)]["threshold"]
                out = [dict(r, pred=(NONE_LABEL if r["conf"] < t else r["pred"]), pred_before_threshold=r["pred"])
                       for r in tst]
                with open(os.path.join(d, f"preds_small_S2_s{s}.jsonl.tmp"), "w") as f:
                    for r in out:
                        f.write(json.dumps(r) + "\n")
                os.replace(os.path.join(d, f"preds_small_S2_s{s}.jsonl.tmp"),
                           os.path.join(d, f"preds_small_S2_s{s}.jsonl"))
                print(f"  S2 seed {s}: threshold {t:.4g} (val acc {th[str(s)]['val_accuracy']:.4f})", flush=True)
            write_meta(d, a.stage, peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    except torch.OutOfMemoryError as e:
        print(f"CUDA OOM: {e}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
