#!/usr/bin/env bash
# Driver for the T1 robustness tests (unattended pipeline). Safe to re-run:
# every stage has a .done_<stage> marker in the output dir and finished work is skipped.
#
#   bash run_stress.sh gpu      GPU stages in order (small_A, small_B, small_C_S1, small_C_S2, train_S3_s0..2)
#   bash run_stress.sh claude   Claude stages in order (claude_A, claude_B, claude_C1)
#   bash run_stress.sh report   wait for the gpu and claude modes to exit, then write the report and commit
#   bash run_stress.sh all      gpu and claude in parallel, then report (use this to resume after a crash)
#   bash run_stress.sh build    only build the frozen test sets
#
# Env: STRESS_DIR (default results/stress), DRY=1 for the tiny dry run (never use with results/stress).
set -u
cd "$(dirname "$(readlink -f "$0")")"
source .venv/bin/activate
# Do not let an IDE / Claude Code session's environment leak into the pipeline or the CLI calls.
for v in $(compgen -e | grep -E '^(CLAUDE|AI_AGENT)'); do [ "$v" = CLAUDE_BIN ] || unset "$v"; done
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODE=${1:-all}
D=${STRESS_DIR:-results/stress}
DRY=${DRY:-0}
if [ "$DRY" = 1 ] && [ "$D" = results/stress ]; then echo "refusing: DRY=1 must not write to results/stress"; exit 2; fi
mkdir -p "$D/logs"
REPORT_OUT=results/t1_robustness.md; [ "$DRY" = 1 ] && REPORT_OUT="$D/t1_robustness.md"
OOM_WAIT=600; [ "$DRY" = 1 ] && OOM_WAIT=10
GPU_WAIT_MAX=21600; GPU_POLL=300
C_STAGES=" small_C_S1 small_C_S2 claude_C1 train_S3_s0 train_S3_s1 train_S3_s2 "

say()   { echo "[$(date '+%F %T')] [$MODE] $*"; }
ts()    { while IFS= read -r l; do printf '%s %s\n' "$(date '+%F %T')" "$l"; done; }
status(){ python stress_common.py status --dir "$D" >/dev/null 2>&1 || true; }
mark()  { python stress_common.py mark --dir "$D" --stage "$1" --state "$2" --reason "${3:-}" --log "$D/logs/$1.log" >/dev/null 2>&1 || true; }
is_done(){ [ -e "$D/.done_$1" ]; }
run_logged(){ local st=$1; shift; "$@" 2>&1 | ts | tee -a "$D/logs/$st.log"; return "${PIPESTATUS[0]}"; }

# Common pre-checks; returns 1 if the stage should not run now.
precheck(){
  local st=$1
  if is_done "$st"; then say "$st already done, skipping"; return 1; fi
  if ! is_done build; then say "$st skipped: build not done"; mark "$st" skipped "dependency build not done"; return 1; fi
  if [[ "$C_STAGES" == *" $st "* ]] && [ -e "$D/test_c_unavailable" ]; then
    mark "$st" skipped "Test C skipped: $(cat "$D/test_c_unavailable")"; say "$st skipped (Test C unavailable)"; return 1
  fi
  return 0
}

# ------------------------------------------------------------------ build
do_build(){
  is_done build && return 0
  mark build running; say "build starting"
  local commit=--commit; [ "$DRY" = 1 ] && commit=""
  if run_logged build python stress_build.py --dir "$D" $commit; then mark build done; say "build done"
  else mark build failed "stress_build.py exited non-zero"; say "build FAILED"; fi
}
ensure_build(){ ( flock 9; do_build ) 9>"$D/.build.lock"; }

# -------------------------------------------------------------------- GPU
wait_gpu(){
  local waited=0 g
  while :; do
    if g=$(python stress_common.py pick_gpu --min_mb 11000 2>/dev/null); then echo "$g"; return 0; fi
    [ $waited -ge $GPU_WAIT_MAX ] && return 1
    say "no GPU with 11 GB free; waiting $((GPU_POLL/60)) min (waited $((waited/60)) min)" >&2
    sleep $GPU_POLL; waited=$((waited+GPU_POLL))
  done
}

gpu_stage(){
  local st=$1; shift
  precheck "$st" || { status; return 0; }
  mark "$st" running
  local attempt=1 extra="" gpu rc
  while :; do
    if ! gpu=$(wait_gpu); then mark "$st" failed "no GPU memory (no GPU had 11 GB free for 6 h)"; say "$st FAILED: no GPU memory"; return 0; fi
    say "$st on GPU $gpu (attempt $attempt${extra:+, $extra})"
    CUDA_VISIBLE_DEVICES=$gpu run_logged "$st" "$@" $extra; rc=$?
    case $rc in
      0) mark "$st" done; say "$st done"; return 0;;
      3) if [ $attempt -eq 1 ]; then say "$st CUDA OOM; waiting $OOM_WAIT s, then retrying with the same settings"
           sleep $OOM_WAIT; attempt=2
         elif [ $attempt -eq 2 ]; then say "$st CUDA OOM again; retrying with the low-memory fallback"
           attempt=3; extra="--low_mem"
         else mark "$st" failed "CUDA OOM after a retry and the low-memory fallback"; say "$st FAILED: OOM"; return 0; fi;;
      4) mark "$st" failed "NaN/inf training loss (learning rate not changed automatically)"; say "$st FAILED: NaN loss"; return 0;;
      *) mark "$st" failed "exit code $rc"; say "$st FAILED: exit $rc"; return 0;;
    esac
  done
}

mode_gpu(){
  ensure_build
  local L="" T=""
  [ "$DRY" = 1 ] && { L="--limit 10 --seeds 0"; T="--dry"; }
  gpu_stage small_A    python stress_eval_small.py --mode A  --dir "$D" --stage small_A $L
  gpu_stage small_B    python stress_eval_small.py --mode B  --dir "$D" --stage small_B $L
  gpu_stage small_C_S1 python stress_eval_small.py --mode S1 --dir "$D" --stage small_C_S1 $L
  gpu_stage small_C_S2 python stress_eval_small.py --mode S2 --dir "$D" --stage small_C_S2 $L
  local seeds="0 1 2"; [ "$DRY" = 1 ] && seeds="0"
  for s in $seeds; do
    gpu_stage "train_S3_s$s" python stress_train_s3.py --seed "$s" --dir "$D" --stage "train_S3_s$s" $T
  done
  if [ "$DRY" = 1 ]; then for s in 1 2; do is_done "train_S3_s$s" || mark "train_S3_s$s" skipped "dry run: seed 0 only"; done; fi
  status
}

# ----------------------------------------------------------------- Claude
claude_stage(){
  local st=$1 set=$2 input=$3 variant=$4 rc
  precheck "$st" || { status; return 0; }
  if [ -e "$D/.model_changed" ]; then mark "$st" failed "aborted: the Claude answering model changed ($(cat "$D/.model_changed"))"; status; return 0; fi
  if [ -e "$D/.claude_flags_missing" ]; then mark "$st" failed "aborted: CLI version changed and required flags are missing"; status; return 0; fi
  if [ "$DRY" = 1 ] && [ "$st" != claude_A ]; then mark "$st" skipped "dry run: only claude_A's first batch"; status; return 0; fi
  mark "$st" running; say "$st starting ($input, prompt $variant)"
  local lim=""; [ "$DRY" = 1 ] && lim="--limit_batches 1"
  run_logged "$st" python baseline_claude_code.py --input "$D/$input" --set "$set" --prompt_variant "$variant" --dir "$D" $lim; rc=$?
  case $rc in
    0)  mark "$st" done; say "$st done";;
    10) mark "$st" failed "aborted: the Claude answering model changed ($(cat "$D/.model_changed" 2>/dev/null))";;
    11) mark "$st" failed "incomplete: Claude call cap (90) reached";;
    12) mark "$st" failed "incomplete: usage/rate limit persisted for 12 h";;
    13) mark "$st" failed "gate batch failed twice (see .gate_$set.json)";;
    14) touch "$D/.claude_flags_missing"; mark "$st" failed "CLI version changed and required flags are missing";;
    *)  mark "$st" failed "exit code $rc";;
  esac
  [ $rc -ne 0 ] && say "$st FAILED (exit $rc)"
  status
}

mode_claude(){
  ensure_build
  claude_stage claude_A  A  typo500.jsonl   base
  claude_stage claude_B  B  merged250.jsonl base
  claude_stage claude_C1 C1 oos_mixed.jsonl none_option
}

# ----------------------------------------------------------------- report
mode_report(){
  sleep $([ "$DRY" = 1 ] && echo 3 || echo 30)
  say "waiting for the gpu and claude modes to exit"
  while python - "$D" <<'EOF'
import sys, stress_common as sc
alive = [m for m in sc.modes_alive(sys.argv[1]) if m in ("gpu", "claude")]
sys.exit(0 if alive else 1)
EOF
  do sleep 60; done
  say "report stage starting"
  mark report running
  if run_logged report python make_robustness_report.py --dir "$D" --out "$REPORT_OUT"; then
    mark report done; say "report written to $REPORT_OUT"
  else
    mark report failed "make_robustness_report.py exited non-zero"; say "report FAILED"
  fi
  status
  if [ "$DRY" != 1 ]; then
    git add run_stress.sh stress_common.py stress_build.py stress_eval_small.py stress_train_s3.py \
            baseline_claude_code.py make_robustness_report.py STRESS_DECISIONS.md \
            "$REPORT_OUT" "$D/STATUS.md" 2>/dev/null
    git add -f "$D"/*.jsonl "$D"/*.json 2>/dev/null
    git commit -q -m "T1 robustness: report and status ($(tail -1 "$D/STATUS.md"))" && say "committed" || say "nothing to commit / commit failed"
  fi
}

# ------------------------------------------------------------------- main
echo $$ > "$D/.mode_${MODE}_running"
trap 'rm -f "$D/.mode_${MODE}_running"; touch "$D/.mode_${MODE}_exited"; status' EXIT
say "mode $MODE starting (dir $D, dry=$DRY, pid $$)"
status
case $MODE in
  build)  ensure_build;;
  gpu)    mode_gpu;;
  claude) mode_claude;;
  report) mode_report;;
  all)    ensure_build
          bash "$0" gpu & bash "$0" claude & wait
          bash "$0" report;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
say "mode $MODE finished"
