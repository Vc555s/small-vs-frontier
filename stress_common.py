#!/usr/bin/env python3
"""Shared helpers for the T1 robustness (stress) pipeline.

Imported by the stage scripts, and used from run_stress.sh as a small CLI:
  python stress_common.py status   --dir D       rewrite D/STATUS.md
  python stress_common.py pick_gpu --min_mb 11000 print the GPU index with most free memory (exit 1 if none)
  python stress_common.py mark     --dir D --stage S --state running|done|failed|skipped [--reason R] [--log F]
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime

NONE_LABEL = "none_of_these"
NONE_SENTENCE = "If a message does not fit any label, answer none_of_these."
CLAUDE_CAP = 90
EXPECTED_MODEL = "claude-opus-5-5"
EXPECTED_CLI_VERSION = "2.1.322"
SMALL_TAG = "Qwen3-4B-Instruct-2507"
SEEDS = [0, 1, 2]

# (stage, kind, depends_on)
STAGES = [
    ("build", "CPU", []),
    ("small_A", "GPU", ["build"]),
    ("small_B", "GPU", ["build"]),
    ("small_C_S1", "GPU", ["build"]),
    ("small_C_S2", "GPU", ["build"]),
    ("claude_A", "Claude", ["build"]),
    ("claude_B", "Claude", ["build"]),
    ("claude_C1", "Claude", ["build"]),
    ("train_S3_s0", "GPU", ["build"]),
    ("train_S3_s1", "GPU", ["build"]),
    ("train_S3_s2", "GPU", ["build"]),
    ("report", "CPU", []),
]
# Stages of run_finish.sh (full-training-set run, figures, reports, push)
FINISH_STAGES = [
    ("full_train", "GPU", []),
    ("full_collect", "CPU", ["full_train"]),
    ("figures", "CPU", []),
    ("reports", "CPU", []),
    ("verify_light", "CPU", ["reports"]),
    ("scan", "CPU", ["reports"]),
    ("summary", "CPU", []),
    ("commit_push", "CPU", ["scan"]),
]
# Stages of run_next.sh (next experiments)
NEXT_STAGES = [
    ("e4a", "GPU", []),
    ("e4b", "GPU", []),
    ("e1_s1", "GPU", []),
    ("e1_s2", "GPU", []),
    ("e2a", "Claude", []),
    ("e2b", "Claude", []),
    ("e3a", "Claude", []),
    ("e3b", "Claude", []),
    ("report", "CPU", []),
    ("scan", "CPU", ["report"]),
    ("commit_push", "CPU", ["scan"]),
]
NEXT_CAP = 140
TEST_C_STAGES = ["small_C_S1", "small_C_S2", "claude_C1", "train_S3_s0", "train_S3_s1", "train_S3_s2"]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extended_prompt(labels):
    """Prompt for C1 / S1 / S3: one extra label line plus one sentence."""
    from t1_common import system_prompt  # lazy: keeps the status CLI light
    return system_prompt(list(labels) + [NONE_LABEL]) + "\n\n" + NONE_SENTENCE


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl_atomic(p, rows):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, p)


def write_json_atomic(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, p)


@contextmanager
def locked(path):
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ------------------------------------------------------------ Claude counter
def claude_calls(d):
    p = os.path.join(d, ".claude_calls")
    try:
        return int(open(p).read().strip() or 0)
    except FileNotFoundError:
        return 0


def take_claude_call(d, cap=None):
    """Reserve one call. Returns the new count, or None if the cap is reached."""
    cap = CLAUDE_CAP if cap is None else cap
    p = os.path.join(d, ".claude_calls")
    with locked(os.path.join(d, ".claude_calls.lock")):
        n = claude_calls(d)
        if n >= cap:
            return None
        with open(p + ".tmp", "w") as f:
            f.write(str(n + 1))
        os.replace(p + ".tmp", p)
        return n + 1


# --------------------------------------------------------------------- GPU
def gpu_free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    res = []
    for line in out.strip().splitlines():
        i, f = [x.strip() for x in line.split(",")]
        res.append((int(i), int(f)))
    return res


def pick_gpu(min_mb):
    gs = sorted(gpu_free_mb(), key=lambda x: -x[1])
    if gs and gs[0][1] >= min_mb:
        return gs[0]
    return None


def write_meta(d, stage, **kw):
    p = os.path.join(d, f".meta_{stage}.json")
    meta = {}
    if os.path.exists(p):
        meta = json.load(open(p))
    for k, v in kw.items():
        if k == "note":
            meta.setdefault("notes", []).append(v)
        elif k == "peak_gb":
            meta["peak_gb"] = max(meta.get("peak_gb", 0), v)
        else:
            meta[k] = v
    write_json_atomic(p, meta)


# ------------------------------------------------------------------ status
def stage_state(d, s):
    for st in ("done", "failed", "skipped", "running"):
        p = os.path.join(d, f".{st}_{s}")
        if os.path.exists(p):
            return st, open(p).read().strip()
    return "pending", ""


def mark(d, stage, state, reason="", log=None):
    os.makedirs(d, exist_ok=True)
    for st in ("done", "failed", "skipped", "running"):
        p = os.path.join(d, f".{st}_{stage}")
        if os.path.exists(p):
            os.remove(p)
    tp = os.path.join(d, f".time_{stage}.json")
    t = json.load(open(tp)) if os.path.exists(tp) else {}
    if state == "running":
        t = {"start": now(), "end": None}
    else:
        t["end"] = now()
        t.setdefault("start", t["end"])
    write_json_atomic(tp, t)
    body = reason
    if log and os.path.exists(log) and state == "failed":
        tail = open(log, errors="replace").read().splitlines()[-30:]
        body += "\n--- last 30 log lines ---\n" + "\n".join(tail)
    with open(os.path.join(d, f".{state}_{stage}"), "w") as f:
        f.write(body)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def modes_alive(d, modes=("gpu", "claude", "report", "all")):
    alive = []
    for m in modes:
        p = os.path.join(d, f".mode_{m}_running")
        if os.path.exists(p):
            try:
                if pid_alive(int(open(p).read().strip())):
                    alive.append(m)
            except ValueError:
                pass
    return alive


def write_status(d, pipeline="stress"):
    stages = {"finish": FINISH_STAGES, "next": NEXT_STAGES}.get(pipeline, STAGES)
    final_stage = "report" if pipeline == "stress" else "commit_push"
    title = {"finish": "T1 finishing pipeline status", "next": "T1 next-experiments pipeline status"}.get(
        pipeline, "T1 robustness pipeline status")
    os.makedirs(d, exist_ok=True)
    with locked(os.path.join(d, ".status.lock")):
        L = [f"# {title}", "", f"Updated {now()} (server time).", ""]
        if os.path.exists(os.path.join(d, ".model_changed")):
            L += ["**WARNING: the Claude answering model changed during the run. Remaining Claude "
                  "stages were aborted.** " + open(os.path.join(d, ".model_changed")).read().strip(), ""]
        L += ["| Stage | Uses | State | Start | End | GPU | Peak GB | Note |",
              "|---|---|---|---|---|---|---|---|"]
        fails = []
        states = {}
        for s, kind, _ in stages:
            st, body = stage_state(d, s)
            states[s] = st
            tp = os.path.join(d, f".time_{s}.json")
            t = json.load(open(tp)) if os.path.exists(tp) else {}
            mp = os.path.join(d, f".meta_{s}.json")
            m = json.load(open(mp)) if os.path.exists(mp) else {}
            note = body.splitlines()[0] if body and st in ("failed", "skipped") else ""
            if m.get("notes"):
                note = (note + " " if note else "") + "; ".join(m["notes"])
            L.append(f"| {s} | {kind} | {st} | {t.get('start') or ''} | {t.get('end') or ''} | "
                     f"{m.get('gpu', '')} | {m.get('peak_gb', '')} | {note} |")
            if st == "failed":
                fails.append((s, body))
        if pipeline in ("stress", "next"):
            L += ["", f"Claude calls used: {claude_calls(d)} of {NEXT_CAP if pipeline == 'next' else CLAUDE_CAP}."]
            wp = os.path.join(d, ".claude_wait")
            if os.path.exists(wp):
                L.append(f"Claude: {open(wp).read().strip()}")
        else:
            L.append("")
        alive = modes_alive(d, ("finish",) if pipeline == "finish" else ("gpu", "claude", "report", "all"))
        final_ready = states[final_stage] in ("done", "failed", "skipped") and not alive
        L.append(f"Pipeline processes alive: {', '.join(alive) if alive else 'none'}.")
        for s, body in fails:
            L += ["", f"## Failure: {s}", "", "```", body, "```"]
        if final_ready:
            gaps = [s for s, _, _ in stages if states[s] != "done"]
            if gaps:
                L += ["", "Stages not done: " + ", ".join(gaps)]
            final = "STATE: FINISHED_WITH_GAPS" if gaps else "STATE: ALL_DONE"
        else:
            if not alive:
                L += ["", "No pipeline process is alive. If this persists, resume with "
                      + {"finish": "`bash run_finish.sh`", "next": "`bash run_next.sh all`"}.get(pipeline, "`bash run_stress.sh all`")
                      + " from the repository root."]
            final = "STATE: RUNNING"
        L += ["", final]
        tmp = os.path.join(d, "STATUS.md.tmp")
        with open(tmp, "w") as f:
            f.write("\n".join(L) + "\n")
        os.replace(tmp, os.path.join(d, "STATUS.md"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["status", "pick_gpu", "mark"])
    p.add_argument("--dir", default="results/stress")
    p.add_argument("--min_mb", type=int, default=11000)
    p.add_argument("--stage")
    p.add_argument("--state")
    p.add_argument("--reason", default="")
    p.add_argument("--log")
    p.add_argument("--pipeline", default="stress", choices=["stress", "finish", "next"])
    a = p.parse_args()
    if a.cmd == "status":
        write_status(a.dir, a.pipeline)
    elif a.cmd == "pick_gpu":
        g = pick_gpu(a.min_mb)
        if g is None:
            raise SystemExit(1)
        print(g[0])
    elif a.cmd == "mark":
        mark(a.dir, a.stage, a.state, a.reason, a.log)
        write_status(a.dir, a.pipeline)


if __name__ == "__main__":
    main()
