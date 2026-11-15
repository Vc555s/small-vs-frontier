#!/usr/bin/env bash
# Baselines: embeddings + logistic regression, fine-tuned ModernBERT, the full training set with
# seeds 1 and 2, Claude Opus with examples in the prompt and Claude Haiku. Resumable: each finished
# stage leaves a .done_<stage> marker in results/baselines/, so a re-run skips it.
#
#   bash run_baselines.sh gpu      embeddings + logistic regression, ModernBERT, full training set seeds 1 and 2
#   bash run_baselines.sh claude   the four Claude runs (Claude CLI, at most 140 calls)
#   bash run_baselines.sh report   wait for the gpu and claude modes to exit, then write the reports
#   bash run_baselines.sh all      gpu and claude in parallel, then report (use this to resume)
#
# Env: VENV (python environment), BASELINES_TEST=1 for a test run in results/baselines_testrun.
set -u
cd "$(dirname "$(readlink -f "$0")")"
[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
for v in $(compgen -e | grep -E '^(CLAUDE|AI_AGENT)'); do [ "$v" = CLAUDE_BIN ] || unset "$v"; done
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODE=${1:-all}
TEST=${BASELINES_TEST:-0}
D=results/baselines; [ "$TEST" = 1 ] && D=results/baselines_testrun
mkdir -p "$D/logs"
OOM_WAIT=600; [ "$TEST" = 1 ] && OOM_WAIT=10
GPU_WAIT_MAX=21600; GPU_POLL=300
CAP=140

say()   { echo "[$(date '+%F %T')] [$MODE] $*"; }
ts()    { while IFS= read -r l; do printf '%s %s\n' "$(date '+%F %T')" "$l"; done; }
status(){ python stress_common.py status --dir "$D" --pipeline baselines >/dev/null 2>&1 || true; }
mark()  { python stress_common.py mark --dir "$D" --stage "$1" --state "$2" --reason "${3:-}" --log "$D/logs/$1.log" --pipeline baselines >/dev/null 2>&1 || true; }
is_done(){ [ -e "$D/.done_$1" ]; }
run_logged(){ local st=$1; shift; "$@" 2>&1 | ts | tee -a "$D/logs/$st.log"; return "${PIPESTATUS[0]}"; }

wait_gpu(){
  local waited=0 g
  while :; do
    if g=$(python stress_common.py pick_gpu --min_mb 11000 2>/dev/null); then echo "$g"; return 0; fi
    [ $waited -ge $GPU_WAIT_MAX ] && return 1
    say "no GPU with 11 GB free; waiting $((GPU_POLL/60)) min (waited $((waited/60)) min)" >&2
    sleep $GPU_POLL; waited=$((waited+GPU_POLL))
  done
}

# gpu_stage NAME command... ; the command may be a shell function
gpu_stage(){
  local st=$1; shift
  if is_done "$st"; then say "$st already done"; return 0; fi
  mark "$st" running
  local attempt=1 gpu rc
  while :; do
    if ! gpu=$(wait_gpu); then mark "$st" failed "no GPU memory (no GPU had 11 GB free for 6 h)"; status; return 0; fi
    say "$st on GPU $gpu (attempt $attempt)"
    CUDA_VISIBLE_DEVICES=$gpu run_logged "$st" "$@"; rc=$?
    case $rc in
      0) mark "$st" done; say "$st done"; status; return 0;;
      3) if [ $attempt -eq 1 ]; then say "$st CUDA OOM; waiting $OOM_WAIT s and retrying once"; sleep $OOM_WAIT; attempt=2
         else mark "$st" failed "CUDA OOM twice"; status; return 0; fi;;
      4) mark "$st" failed "NaN/inf training loss (settings not changed)"; status; return 0;;
      *) mark "$st" failed "exit code $rc"; status; return 0;;
    esac
  done
}

full_seed(){
  local s=$1
  local dry=""; [ "$TEST" = 1 ] && dry=1
  FULL_DATA_TEST=$dry python finetune_qwen_full.py --seed "$s" --out "$D/full_data_s$s" --dest "$D" || return $?
  FULL_DATA_TEST=$dry python finetune_qwen_full.py --collect --seed "$s" --out "$D/full_data_s$s" --dest "$D" --log "$D/logs/full_s$s.log"
}

mode_gpu(){
  local T="" B=""; [ "$TEST" = 1 ] && { T="--test"; B="--budgets 100 --seeds 0"; }
  gpu_stage embeddings_logreg python baseline_embeddings_logreg.py --dir "$D" $T $B
  local B2=""; [ "$TEST" = 1 ] && B2="--budgets 2000 --seeds 0"
  gpu_stage modernbert python baseline_modernbert.py --dir "$D" $T $B2
  gpu_stage full_s1 full_seed 1
  gpu_stage full_s2 full_seed 2
}

claude_stage(){
  local st=$1 set=$2; shift 2
  if is_done "$st"; then say "$st already done"; return 0; fi
  if [ -e "$D/.claude_flags_missing" ]; then mark "$st" failed "CLI version changed and required flags are missing"; status; return 0; fi
  mark "$st" running; say "$st starting"
  local lim=""; [ "$TEST" = 1 ] && lim="--limit_batches 1"
  run_logged "$st" python claude_baseline.py --input results/baselines/clean500.jsonl --set "$set" --dir "$D" \
    --cap $CAP --timeout 600 $lim "$@"; local rc=$?
  case $rc in
    0)  mark "$st" done; say "$st done";;
    10) mark "$st" failed "stopped: the CLI answered with a different model ($(cat "$D/.model_changed" 2>/dev/null))"
        rm -f "$D/.model_changed";;
    11) mark "$st" failed "incomplete: Claude call cap ($CAP) reached";;
    12) mark "$st" failed "incomplete: usage/rate limit persisted for 12 h";;
    13) mark "$st" failed "gate batch failed twice (see .gate_$set.json)";;
    14) touch "$D/.claude_flags_missing"; mark "$st" failed "CLI version changed and required flags are missing";;
    *)  mark "$st" failed "exit code $rc";;
  esac
  status
}

mode_claude(){
  local ONE=results/baselines/icl_one_per_class.json RET=results/baselines/icl_retrieved
  claude_stage claude_opus_one_example opus_one_example_per_intent --examples_file "$ONE"
  claude_stage claude_opus_retrieved opus_retrieved_examples --examples_dir "$RET"
  claude_stage claude_haiku_zero_shot haiku_zero_shot --model claude-haiku-5-5 --expected_model claude-haiku-5-5
  claude_stage claude_haiku_one_example haiku_one_example_per_intent --model claude-haiku-5-5 --expected_model claude-haiku-5-5 --examples_file "$ONE"
}

cpu_stage(){
  local st=$1; shift; local deps=()
  while [ "$1" != "--" ]; do deps+=("$1"); shift; done; shift
  if is_done "$st" && [ "$st" != report ]; then say "$st already done"; return 0; fi
  for dep in "${deps[@]}"; do
    if ! is_done "$dep"; then mark "$st" skipped "dependency $dep not done"; status; return 0; fi
  done
  mark "$st" running; say "$st starting"
  if run_logged "$st" "$@"; then mark "$st" done; say "$st done"; else mark "$st" failed "exit code $?"; say "$st FAILED"; fi
  status
}

reports(){
  python make_baselines_report.py --dir "$D" --out "$([ "$TEST" = 1 ] && echo "$D/baselines_report.md" || echo results/baselines_report.md)" || return 1
  [ "$TEST" = 1 ] && return 0
  python make_main_report.py
}

mode_report(){
  sleep $([ "$TEST" = 1 ] && echo 3 || echo 30)
  say "waiting for the gpu and claude modes to exit"
  while python - "$D" <<'EOF'
import sys, stress_common as sc
sys.exit(0 if [m for m in sc.modes_alive(sys.argv[1]) if m in ("gpu", "claude")] else 1)
EOF
  do sleep 60; done
  cpu_stage report -- reports
}

echo $$ > "$D/.mode_${MODE}_running"
trap 'rm -f "$D/.mode_${MODE}_running"; touch "$D/.mode_${MODE}_exited"; status' EXIT
say "mode $MODE starting (dir $D, test=$TEST, pid $$)"
status
case $MODE in
  gpu)    mode_gpu;;
  claude) mode_claude;;
  report) mode_report;;
  all)    bash "$0" gpu & bash "$0" claude & wait
          bash "$0" report;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
say "mode $MODE finished"
