#!/usr/bin/env python3
"""
T1 baseline: Claude via the Claude Code CLI (Pro subscription), Banking77 routing.

NOT an API baseline. One Claude model, whichever the Pro plan serves, called through
the `claude` CLI in print mode. The rules are summarised in results/t1_report.md.

Each batch of 25 test messages is sent to a fresh `claude -p` process whose cwd is a
new empty temp dir, with all tools disabled, no CLAUDE.md / memory / skills / hooks /
MCP servers, and no session persistence. Gold labels are never in the prompt; they
are joined to predictions here, after the subprocess returns.

Usage:
  python baseline_claude_code.py --limit_batches 1     # one-batch test
  python baseline_claude_code.py                       # all remaining batches, then score
  python baseline_claude_code.py --score_only          # just (re)write the report
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date

from exp1_routing import load_data, norm, system_prompt

BATCH = 25
SAMPLE_N = 500
OUT = "results"
SAMPLE_FILE = os.path.join(OUT, "sample500.json")
ORDER_FILE = os.path.join(OUT, "sample500_batch_order.json")
ORDER_SEED = 456
PREDS_FILE = os.path.join(OUT, "preds_claude_code_sample500.jsonl")
RAW_DIR = os.path.join(OUT, "claude_code_raw")
REPORT = os.path.join(OUT, "t1_claude_vs_small_sample500.md")
SMALL_TAG = "Qwen3-4B-Instruct-2507"
SMALL_BUDGETS = [0, 100, 500, 2000]
SMALL_SEEDS = [0, 1, 2]
LABEL = "Claude via Claude Code CLI (Pro subscription)"

# Written once (rule 3/6): minimum extra wording for batched JSON output.
BATCH_ADDITION = ("\n\nYou will receive 25 numbered customer messages. Return only a JSON "
                  "array of exactly 25 strings, where item i is the label for message i, "
                  "in order. Use only labels from the list. No explanations, no markdown.")


def find_claude(explicit):
    if explicit:
        return explicit
    for c in [os.environ.get("CLAUDE_BIN"), shutil.which("claude")]:
        if c:
            return c
    # Fallback: the binary bundled with the editor extension (no `claude` on PATH here).
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.*-server/extensions/anthropic.claude-code-*/resources/native-binary/claude")))
    if hits:
        return hits[-1]
    raise SystemExit("No claude CLI found. Pass --claude_bin or set CLAUDE_BIN.")


def cli_cmd(claude, sys_p, model):
    cmd = [claude, "-p",
           "--output-format", "json",
           "--system-prompt", sys_p,        # replaces the default Claude Code prompt
           "--tools", "",                   # no tools at all -> exactly one turn possible
           "--safe-mode",                   # no CLAUDE.md, memory, skills, hooks, plugins
           "--strict-mcp-config",           # no MCP servers
           "--disable-slash-commands",
           "--no-session-persistence",
           "--permission-prompts", "none"]
    if model:
        cmd += ["--model", model]
    return cmd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit_batches", type=int, default=0, help="run only the first N batches")
    p.add_argument("--claude_bin", default="")
    p.add_argument("--model", default="", help="leave empty to use whatever the plan serves")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--pause", type=float, default=5.0)
    p.add_argument("--score_only", action="store_true")
    # Robustness (stress) sets; see STRESS_DECISIONS.md.
    p.add_argument("--input", default="", help="stress-set JSONL (id, text, gold, ...)")
    p.add_argument("--prompt_variant", default="base", choices=["base", "none_option"])
    p.add_argument("--set", default="", help="set name used in output file names, e.g. A, B, C1")
    p.add_argument("--dir", default="results/stress", help="output dir for stress sets")
    return p.parse_args()


# ------------------------------------------------------------------ sample
def get_sample(test):
    if os.path.exists(SAMPLE_FILE):
        with open(SAMPLE_FILE) as f:
            idx = json.load(f)["indices"]
    else:
        idx = sorted(random.Random(123).sample(range(len(test)), SAMPLE_N))
        with open(SAMPLE_FILE, "w") as f:
            json.dump({"seed": 123, "n": SAMPLE_N, "test_size": len(test),
                       "indices": idx}, f)
    assert len(idx) == SAMPLE_N
    return idx


def get_batch_order(idx):
    """Banking77's test set is sorted by label, so batching the sorted indices puts
    long same-label runs in each call. Shuffle once (seed 456) so each batch mixes
    intents. Same 500 examples; only which messages share a call changes."""
    if os.path.exists(ORDER_FILE):
        with open(ORDER_FILE) as f:
            order = json.load(f)["order"]
    else:
        order = idx[:]
        random.Random(ORDER_SEED).shuffle(order)
        with open(ORDER_FILE, "w") as f:
            json.dump({"seed": ORDER_SEED, "source": "sample500.json", "order": order}, f)
    assert sorted(order) == sorted(idx), "batch order must be a permutation of the sample"
    return order


# ------------------------------------------------------------------- calls
def main_model(meta):
    """The model whose token counts match the top-level `usage` (the answering call).
    Other entries in modelUsage are the CLI's own side calls."""
    mu = meta.get("modelUsage") or {}
    u = meta.get("usage") or {}
    for name, m in mu.items():
        if (m.get("outputTokens") == u.get("output_tokens")
                and m.get("cacheCreationInputTokens") == u.get("cache_creation_input_tokens")):
            return name
    return max(mu, key=lambda k: mu[k].get("outputTokens", 0)) if mu else None


def is_rate_limited(meta, stdout, stderr):
    text = " ".join([str(meta.get("result", "")), str(meta.get("api_error_status", "")),
                     stdout[-2000:], stderr[-2000:]]).lower()
    return (meta.get("api_error_status") == 429
            or any(k in text for k in ["rate limit", "rate_limit", "usage limit",
                                        "limit reached", "limit will reset"]))


def parse_array(text, n=BATCH):
    """First JSON array in the text; valid only if n strings."""
    start = text.find("[")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        arr = json.loads(text[start:j + 1])
                    except json.JSONDecodeError:
                        break
                    if (isinstance(arr, list) and len(arr) == n
                            and all(isinstance(x, str) for x in arr)):
                        return arr, None
                    return None, f"array has {len(arr) if isinstance(arr, list) else '?'} items or non-strings"
        start = text.find("[", start + 1)
    return None, "no JSON array found"


def call_batch(cmd, user_msg, timeout, b, attempt):
    tmp = tempfile.mkdtemp(prefix="cc_t1_")
    before = os.listdir(tmp)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, input=user_msg, capture_output=True, text=True,
                              cwd=tmp, timeout=timeout)
        stdout, stderr, rc, err = proc.stdout, proc.stderr, proc.returncode, None
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, err = None, f"timeout after {timeout}s"
    wall = time.time() - t0
    after = os.listdir(tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    meta = {}
    if stdout.strip():
        try:
            meta = json.loads(stdout)
        except json.JSONDecodeError:
            err = err or "CLI stdout is not JSON"
    raw_record = {"batch": b, "attempt": attempt, "cmd": cmd[:1] + ["<args below>"],
                  "args": cmd[1:], "returncode": rc, "wall_seconds": round(wall, 2),
                  "tmp_dir": tmp, "tmp_before": before, "tmp_after": after,
                  "stdout_json": meta or None, "stdout_text": None if meta else stdout,
                  "stderr": stderr}
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, f"batch_{b:02d}_attempt_{attempt}.json"), "w") as f:
        json.dump(raw_record, f, indent=1)

    if is_rate_limited(meta, stdout, stderr):
        return None, meta, wall, "RATE_LIMIT"
    if err:
        return None, meta, wall, err
    if meta.get("is_error") or meta.get("subtype") != "success":
        return None, meta, wall, f"CLI error: subtype={meta.get('subtype')} result={str(meta.get('result'))[:200]}"
    if meta.get("num_turns") != 1:
        return None, meta, wall, f"num_turns={meta.get('num_turns')} (expected 1)"
    arr, perr = parse_array(meta.get("result") or "")
    return arr, meta, wall, perr


def done_batches():
    done = set()
    if os.path.exists(PREDS_FILE):
        counts = {}
        with open(PREDS_FILE) as f:
            for line in f:
                r = json.loads(line)
                counts[r["batch"]] = counts.get(r["batch"], 0) + 1
        done = {b for b, c in counts.items() if c == BATCH}
    return done


def run(args, test, labels, idx):
    claude = find_claude(args.claude_bin)
    sys_p = system_prompt(labels) + BATCH_ADDITION
    cmd = cli_cmd(claude, sys_p, args.model)
    valid = {norm(l): l for l in labels}
    order = get_batch_order(idx)
    batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    todo = [b for b in range(len(batches)) if b not in done_batches()]
    if args.limit_batches:
        todo = [b for b in todo if b < args.limit_batches]
    print(f"claude={claude}\nbatches total={len(batches)} done={len(batches) - len([b for b in range(len(batches)) if b not in done_batches()])} to_run={todo}")

    for k, b in enumerate(todo):
        bidx = batches[b]
        # Message texts only: no gold labels reach the subprocess.
        user_msg = "\n".join(f"{i + 1}. {' '.join(test[j][0].split())}"
                             for i, j in enumerate(bidx))
        arr = meta = err = None
        for attempt in (1, 2):
            arr, meta, wall, err = call_batch(cmd, user_msg, args.timeout, b, attempt)
            if err == "RATE_LIMIT":
                print(f"Rate/usage limit hit on batch {b}. Stopping cleanly. "
                      f"Completed batches: {sorted(done_batches())}. Re-run later to resume.")
                return
            print(f"batch {b:02d} attempt {attempt}: wall {wall:.1f}s "
                  f"model={main_model(meta) if meta else None} "
                  f"{'OK' if arr else 'FAILED: ' + str(err)}", flush=True)
            if arr:
                break
            if attempt == 1:
                time.sleep(args.pause)
        model = main_model(meta) if meta else None
        with open(PREDS_FILE, "a") as f:
            for i, j in enumerate(bidx):
                raw = arr[i] if arr else None
                f.write(json.dumps({
                    "idx": j, "text": test[j][0], "gold": test[j][1], "raw": raw,
                    "pred": valid.get(norm(raw)) if raw is not None else None,
                    "batch": b, "attempt": attempt, "model": model,
                    "wall_seconds": round(wall, 2), "error": None if arr else err}) + "\n")
        if k < len(todo) - 1:
            time.sleep(args.pause)


# ----------------------------------------------------------------- scoring
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def metrics(golds, preds, labels):
    from sklearn.metrics import f1_score
    n = len(golds)
    acc = sum(g == p for g, p in zip(golds, preds)) / n
    valid = sum(p is not None for p in preds) / n
    f1 = f1_score(golds, [p if p is not None else "__invalid__" for p in preds],
                  labels=labels, average="macro", zero_division=0)
    return acc, f1, valid


def mean_std(xs):
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return m, s


def score(test, labels, idx):
    rows = [json.loads(l) for l in open(PREDS_FILE)] if os.path.exists(PREDS_FILE) else []
    by_idx = {r["idx"]: r for r in rows}
    batches = sorted({r["batch"] for r in rows})
    retried = len({r["batch"] for r in rows if r["attempt"] == 2})
    failed = len({r["batch"] for r in rows if r["error"]})
    models = sorted({r["model"] for r in rows if r["model"]})
    have = [i for i in idx if i in by_idx]

    lines = [f"# T1 routing on a fixed 500-example Banking77 sample: {LABEL} vs fine-tuned small model",
             "", f"Run date: {date.today().isoformat()}. "
             f"Sample: `results/sample500.json` (500 test indices, `random.Random(123)`).", ""]
    if len(have) < SAMPLE_N:
        lines += [f"**Incomplete: Claude predictions exist for {len(have)}/{SAMPLE_N} examples "
                  f"({len(batches)}/20 batches). Numbers below cover those only.**", ""]

    table = ["| System | Label budget | Accuracy on sample | Macro-F1 | Valid-label rate | Runs |",
             "|---|---|---|---|---|---|"]
    if have:
        g = [by_idx[i]["gold"] for i in have]
        p = [by_idx[i]["pred"] for i in have]
        acc, f1, v = metrics(g, p, labels)
        k = sum(a == b for a, b in zip(g, p))
        lo, hi = wilson(k, len(have))
        table.append(f"| {LABEL} | n/a (prompted) | {acc:.3f} (95% Wilson CI {lo:.3f}–{hi:.3f}) "
                     f"| {f1:.3f} | {v:.3f} | 1 |")

    missing = []
    for n in SMALL_BUDGETS:
        accs, f1s, vals = [], [], []
        for s in SMALL_SEEDS:
            fn = os.path.join(OUT, f"preds_{SMALL_TAG}_n{n}_s{s}.jsonl")
            if not os.path.exists(fn):
                missing.append(os.path.basename(fn))
                continue
            sp = [json.loads(l) for l in open(fn)]
            assert len(sp) == len(test), f"{fn}: {len(sp)} lines vs test {len(test)}"
            for i in have or idx:
                assert sp[i]["text"] == test[i][0], f"{fn}: line {i} text mismatch"
            use = have or idx
            a, f, v = metrics([sp[i]["gold"] for i in use], [sp[i]["pred"] for i in use], labels)
            accs.append(a); f1s.append(f); vals.append(v)
        if not accs:
            continue
        ma, sa = mean_std(accs); mf, sf = mean_std(f1s); mv, _ = mean_std(vals)
        pm = lambda m, s: f"{m:.3f}" + (f" ± {s:.3f}" if len(accs) > 1 else "")
        name = f"{SMALL_TAG} " + ("untuned" if n == 0 else "LoRA")
        table.append(f"| {name} | {n} | {pm(ma, sa)} | {pm(mf, sf)} | {mv:.3f} | {len(accs)} seed(s) |")

    lines += table + ["", "Small-model numbers are mean ± sample std over seeds. "
                      "Budget 0 is deterministic (greedy) and was only run for seed 0.", ""]
    if missing:
        lines += ["Small-model prediction files not found (run not finished or not run): "
                  + ", ".join(f"`{m}`" for m in missing if "_n0_s" not in m or m.endswith("_s0.jsonl")), ""]

    try:
        ver = subprocess.run([find_claude(""), "--version"], capture_output=True,
                             text=True, timeout=30, cwd=tempfile.gettempdir()).stdout.strip()
    except Exception as e:  # noqa: BLE001
        ver = f"unknown ({e})"
    lines += ["## Claude setup", "",
              f"- Model reported by the CLI (answering call): {', '.join(f'`{m}`' for m in models) or 'n/a'}",
              f"- CLI version: `{ver}`",
              "- Flags: `-p --output-format json --system-prompt <ours> --tools \"\" --safe-mode "
              "--strict-mcp-config --disable-slash-commands --no-session-persistence --permission-prompts none`",
              "- Each call ran in a fresh, empty temp directory; message texts only were sent.",
              f"- **Batch composition (changed after the one-batch test):** the 500 sample indices were "
              f"shuffled once with `random.Random({ORDER_SEED})` and split into 20 batches of 25 "
              f"(order saved in `results/sample500_batch_order.json`). The original plan batched them "
              "in index order, but the Banking77 test set is sorted by label, so each index-order batch "
              "held only 4–7 intents in long same-label runs. That lets Claude infer labels from "
              "neighbouring messages, which the one-at-a-time small model cannot. The discarded "
              "first test batch (index order) scored 24/25, but that number is not used anywhere. "
              "That batch's predictions and raw output were deleted and all 20 batches were rerun "
              "with the shuffled order. The prompt wording and scoring were not changed.",
              f"- Batches: {len(batches)} run, {retried} retried, {failed} failed "
              f"(failed batches count as 25 wrong answers).", "",
              "## Known differences from the small-model setup", "",
              "- Claude saw **25 messages per call**; the small model saw one at a time.",
              "- The Claude Code CLI wraps our system prompt in some text of its own, even with "
              "`--system-prompt` (a 15-character test prompt was billed as ~789 prompt tokens). "
              "The CLI also makes a small side call to `claude-haiku-5-5`, which does not produce the answers.",
              "- **Temperature could not be set.** The small model used greedy decoding.",
              "- The model is whichever one the Pro plan served (recorded above), not a chosen API model.",
              "- Wall time per call includes CLI start-up and network; it is not model latency "
              "and is not comparable to local latency.",
              "- This is one Claude model through one CLI. It is not an API baseline and says nothing about other frontier models.",
              ""]
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ------------------------------------------------------------ stress sets
REQUIRED_FLAGS = ["--print", "--output-format", "--system-prompt", "--tools", "--safe-mode",
                  "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
                  "--permission-prompts"]
TRANSIENT_WAITS = [30, 120, 600]
LIMIT_SLEEP = 1800
LIMIT_WINDOW = 12 * 3600
EXIT = {"model_changed": 10, "cap": 11, "limit_window": 12, "gate": 13, "flags": 14}


def invoke(cmd, user_msg, timeout, raw_path, meta_extra):
    tmp = tempfile.mkdtemp(prefix="cc_stress_")
    before = os.listdir(tmp)
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(cmd, input=user_msg, capture_output=True, text=True, cwd=tmp,
                              timeout=timeout)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, timed_out = None, True
    wall = time.time() - t0
    after = os.listdir(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    meta = None
    try:
        meta = json.loads(stdout) if stdout.strip() else None
    except json.JSONDecodeError:
        meta = None
    rec = dict(meta_extra, returncode=rc, timed_out=timed_out, wall_seconds=round(wall, 2),
               tmp_dir=tmp, tmp_before=before, tmp_after=after, stdout_json=meta,
               stdout_text=None if meta else stdout, stderr=stderr,
               args=cmd[1:])
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(rec, f, indent=1)
    return {"rc": rc, "timed_out": timed_out, "meta": meta or {}, "parsed": meta is not None,
            "stdout": stdout, "stderr": stderr, "wall": wall, "tmp_ok": before == [] and after == []}


def classify(r):
    m = r["meta"]
    clean = r["rc"] == 0 and r["parsed"] and not m.get("is_error") and m.get("subtype") == "success"
    if clean:
        return "ok"
    if is_rate_limited(m, r["stdout"], r["stderr"]):
        return "limit"
    return "transient"


def run_stress_set(args):
    import stress_common as sc
    d, setname = args.dir, args.set
    os.makedirs(d, exist_ok=True)
    items = [json.loads(l) for l in open(args.input) if l.strip()]
    _, _, labels = load_data("mteb/banking77")
    allowed = list(labels) + ([sc.NONE_LABEL] if args.prompt_variant == "none_option" else [])
    valid = {norm(l): l for l in allowed}
    base_sys = sc.extended_prompt(labels) if args.prompt_variant == "none_option" else system_prompt(labels)
    preds_path = os.path.join(d, f"preds_claude_{setname}.jsonl")
    raw_dir = os.path.join(d, "claude_raw", setname)
    gate_path = os.path.join(d, f".gate_{setname}.json")
    wait_path = os.path.join(d, ".claude_wait")

    def log(msg):
        print(f"[{sc.now()}] claude {setname}: {msg}", flush=True)

    # Resolve the CLI now (each stage), check version / flags.
    claude = find_claude(args.claude_bin)
    ver = subprocess.run([claude, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    log(f"CLI {claude} version {ver!r}")
    if not ver.startswith(sc.EXPECTED_CLI_VERSION):
        helptext = subprocess.run([claude, "--help"], capture_output=True, text=True, timeout=60).stdout
        missing = [f for f in REQUIRED_FLAGS if f not in helptext]
        if missing:
            log(f"CLI version changed and flags are missing: {missing}")
            sys.exit(EXIT["flags"])
        log("CLI version differs from 2.1.322 but every required flag is present")

    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    done = set()
    if os.path.exists(preds_path):
        cnt = {}
        for l in open(preds_path):
            r = json.loads(l)
            cnt[r["batch"]] = cnt.get(r["batch"], 0) + 1
        done = {b for b, c in cnt.items() if c == len(batches[b])}
    gate_passed = os.path.exists(gate_path) and json.load(open(gate_path)).get("passed")
    todo = [b for b in range(len(batches)) if b not in done]
    if args.limit_batches:
        todo = [b for b in todo if b < args.limit_batches]
    log(f"{len(batches)} batches, {len(done)} done, to run {todo}; gate passed: {bool(gate_passed)}")
    waited = 0

    for k, b in enumerate(todo):
        batch = batches[b]
        n = len(batch)
        addition = BATCH_ADDITION if n == BATCH else BATCH_ADDITION.replace("25", str(n))
        cmd = cli_cmd(claude, base_sys + addition, args.model)
        user_msg = "\n".join(f"{i + 1}. {' '.join(r['text'].split())}" for i, r in enumerate(batch))
        is_gate = (b == 0 and not gate_passed)
        transient_used, malformed_used, gate_retry_used = 0, False, False
        attempt = 0
        while True:
            c = sc.take_claude_call(d)
            if c is None:
                log(f"call cap {sc.CLAUDE_CAP} reached before batch {b}")
                sys.exit(EXIT["cap"])
            attempt += 1
            r = invoke(cmd, user_msg, args.timeout,
                       os.path.join(raw_dir, f"batch_{b:02d}_attempt_{attempt}.json"),
                       {"set": setname, "batch": b, "attempt": attempt, "call_number": c,
                        "prompt_variant": args.prompt_variant, "cli_version": ver})
            kind = classify(r)
            m = r["meta"]
            model = main_model(m) if m else None
            log(f"batch {b:02d} attempt {attempt} (call {c}): {kind}, wall {r['wall']:.1f}s, model {model}")
            if kind == "limit":
                if waited + LIMIT_SLEEP > LIMIT_WINDOW:
                    log("usage/rate limit persisted for the whole 12 h window; set incomplete")
                    if os.path.exists(wait_path):
                        os.remove(wait_path)
                    sys.exit(EXIT["limit_window"])
                with open(wait_path, "w") as f:
                    f.write(f"set {setname} paused at batch {b} by a usage/rate limit since {sc.now()}; "
                            f"retrying every 30 min (waited {waited // 60} min so far, max 12 h)")
                sc.write_status(d)
                time.sleep(LIMIT_SLEEP)
                waited += LIMIT_SLEEP
                continue
            if os.path.exists(wait_path):
                os.remove(wait_path)
                sc.write_status(d)
            if kind == "transient":
                if transient_used < len(TRANSIENT_WAITS):
                    w = TRANSIENT_WAITS[transient_used]
                    transient_used += 1
                    log(f"transient error (rc={r['rc']}, timeout={r['timed_out']}); retry in {w}s")
                    time.sleep(w)
                    continue
                err = f"transient errors persisted (rc={r['rc']}, timeout={r['timed_out']})"
                arr = None
            else:
                if model != sc.EXPECTED_MODEL:
                    with open(os.path.join(d, ".model_changed"), "w") as f:
                        f.write(f"set {setname} batch {b}: answering model was {model!r}, "
                                f"expected {sc.EXPECTED_MODEL!r} ({sc.now()})")
                    log(f"answering model changed to {model}; aborting Claude stages")
                    sys.exit(EXIT["model_changed"])
                arr, err = parse_array(m.get("result") or "", n)
                if m.get("num_turns") != 1:
                    arr, err = None, f"num_turns={m.get('num_turns')}"
                if not r["tmp_ok"]:
                    arr, err = None, "temp dir was not empty"
            if is_gate:
                nvalid = sum(valid.get(norm(x)) is not None for x in arr) if arr else 0
                ok = (kind == "ok" and arr is not None and nvalid >= n - 2)
                gate = {"passed": ok, "batch": b, "attempt": attempt, "valid_labels": nvalid, "n": n,
                        "model": model, "num_turns": m.get("num_turns"), "returncode": r["rc"],
                        "error": err, "time": sc.now()}
                if not ok:
                    if not gate_retry_used:
                        gate_retry_used = True
                        log(f"gate FAILED ({gate}); retrying once in 60 s")
                        time.sleep(60)
                        continue
                    with open(gate_path, "w") as f:
                        json.dump(gate, f, indent=1)
                    log(f"gate FAILED twice ({gate}); set failed")
                    sys.exit(EXIT["gate"])
                with open(gate_path, "w") as f:
                    json.dump(gate, f, indent=1)
                gate_passed = True
                log(f"gate passed: {nvalid}/{n} allowed labels")
            elif arr is None and kind == "ok" and not malformed_used:
                malformed_used = True
                log(f"malformed batch ({err}); retrying once")
                time.sleep(args.pause)
                continue
            break
        with open(preds_path, "a") as f:
            for i, it in enumerate(batch):
                raw = arr[i] if arr else None
                rec = {k2: v for k2, v in it.items() if k2 != "orig_text"}
                rec.update({"raw": raw, "pred": valid.get(norm(raw)) if raw is not None else None,
                            "batch": b, "attempt": attempt, "model": model,
                            "wall_seconds": round(r["wall"], 2), "error": None if arr else err})
                f.write(json.dumps(rec) + "\n")
        if k < len(todo) - 1:
            time.sleep(args.pause)
    log("finished")


def main():
    args = get_args()
    if args.input:
        run_stress_set(args)
        return
    os.makedirs(OUT, exist_ok=True)
    _, test, labels = load_data("mteb/banking77")
    assert len(test) == 3076 and len(labels) == 77, (len(test), len(labels))
    idx = get_sample(test)
    if not args.score_only:
        run(args, test, labels, idx)
    if args.score_only or (not args.limit_batches and len(done_batches()) == SAMPLE_N // BATCH):
        score(test, labels, idx)


if __name__ == "__main__":
    main()
