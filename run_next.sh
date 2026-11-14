#!/usr/bin/env bash
# Next experiments for T1 (E1-E6). Safe to re-run: each stage has a .done_<stage> marker in
# results/next/ and finished work is skipped.
#
#   bash run_next.sh gpu      E4a, E4b, then E1 seeds 1 and 2
#   bash run_next.sh claude   E2a, E2b, E3a, E3b (Claude CLI, at most 140 calls)
#   bash run_next.sh report   wait for gpu and claude to exit, then report, scan, commit and push the branch
#   bash run_next.sh all      gpu and claude in parallel, then report (use this to resume)
#
# Env: VENV (python environment), SCAN_SCRIPT (private-detail scanner with --check; without it the
# scan fails and nothing is pushed), NEXT_TEST=1 for a test run in results/next_testrun (no commit).
set -u
cd "$(dirname "$(readlink -f "$0")")"
[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
for v in $(compgen -e | grep -E '^(CLAUDE|AI_AGENT)'); do [ "$v" = CLAUDE_BIN ] || unset "$v"; done
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODE=${1:-all}
TEST=${NEXT_TEST:-0}
D=results/next; [ "$TEST" = 1 ] && D=results/next_testrun
mkdir -p "$D/logs"
OOM_WAIT=600; [ "$TEST" = 1 ] && OOM_WAIT=10
GPU_WAIT_MAX=21600; GPU_POLL=300
CAP=140

say()   { echo "[$(date '+%F %T')] [$MODE] $*"; }
ts()    { while IFS= read -r l; do printf '%s %s\n' "$(date '+%F %T')" "$l"; done; }
status(){ python stress_common.py status --dir "$D" --pipeline next >/dev/null 2>&1 || true; }
mark()  { python stress_common.py mark --dir "$D" --stage "$1" --state "$2" --reason "${3:-}" --log "$D/logs/$1.log" --pipeline next >/dev/null 2>&1 || true; }
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

e1_seed(){
  local s=$1
  local dry=""; [ "$TEST" = 1 ] && dry=1
  FINISH_DRY=$dry python full_data.py --seed "$s" --out "$D/full_data_s$s" --dest "$D" || return $?
  FINISH_DRY=$dry python full_data.py --collect --seed "$s" --out "$D/full_data_s$s" --dest "$D" --log "$D/logs/e1_s$s.log"
}

mode_gpu(){
  local T="" B=""; [ "$TEST" = 1 ] && { T="--test"; B="--budgets 100 --seeds 0"; }
  gpu_stage e4a python next_e4a.py --dir "$D" $T $B
  local B2=""; [ "$TEST" = 1 ] && B2="--budgets 2000 --seeds 0"
  gpu_stage e4b python next_e4b.py --dir "$D" $T $B2
  gpu_stage e1_s1 e1_seed 1
  gpu_stage e1_s2 e1_seed 2
}

claude_stage(){
  local st=$1 set=$2; shift 2
  if is_done "$st"; then say "$st already done"; return 0; fi
  if [ -e "$D/.claude_flags_missing" ]; then mark "$st" failed "CLI version changed and required flags are missing"; status; return 0; fi
  mark "$st" running; say "$st starting"
  local lim=""; [ "$TEST" = 1 ] && lim="--limit_batches 1"
  run_logged "$st" python baseline_claude_code.py --input results/next/clean500.jsonl --set "$set" --dir "$D" \
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
  local ONE=results/next/icl_one_per_class.json RET=results/next/icl_retrieved
  claude_stage e2a E2a --examples_file "$ONE"
  claude_stage e2b E2b --examples_dir "$RET"
  claude_stage e3a E3a --model claude-haiku-5-5 --expected_model claude-haiku-5-5
  claude_stage e3b E3b --model claude-haiku-5-5 --expected_model claude-haiku-5-5 --examples_file "$ONE"
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
  python make_baselines_report.py --next "$D" --out "$([ "$TEST" = 1 ] && echo "$D/t1_baselines.md" || echo results/t1_baselines.md)" || return 1
  [ "$TEST" = 1 ] && return 0
  python make_t1_report.py
}

scan(){
  [ -n "${SCAN_SCRIPT:-}" ] && [ -f "$SCAN_SCRIPT" ] || { echo "SCAN_SCRIPT not set; cannot scan"; return 1; }
  python "$SCAN_SCRIPT" . --check
}

commit_push(){
  if [ "$TEST" = 1 ]; then echo "test run: would commit"; git status --short; return 0; fi
  [ "$(git branch --show-current)" = next-experiments ] || { echo "not on next-experiments; refusing"; return 1; }
  commit(){ local msg=$1; shift; git add -- "$@" 2>/dev/null; if ! git diff --cached --quiet; then git commit -q -m "$msg" && echo "committed: $msg"; fi; }
  commit "Next experiments: scripts, frozen example lists and pipeline" \
         next_build.py next_e4a.py next_e4b.py make_baselines_report.py run_next.sh full_data.py \
         baseline_claude_code.py stress_common.py make_t1_report.py requirements.txt REPRODUCE.md .gitignore \
         results/next/clean500.jsonl results/next/icl_one_per_class.json results/next/icl_retrieved results/next/deviations.json
  commit "E4: embedding + logistic regression and fine-tuned ModernBERT baselines" \
         results/next/preds_bge-small-en-v1.5-logreg_* results/next/result_bge-small-en-v1.5-logreg.json results/next/latency_e4a.json \
         results/next/preds_ModernBERT-base-ft_* results/next/result_ModernBERT-base-ft.json results/next/latency_e4b.json
  commit "E1: full training set, seeds 1 and 2" \
         results/next/preds_Qwen3-4B-Instruct-2507_n9993_s* results/next/exp1_Qwen3-4B-Instruct-2507_n9993_s* \
         results/next/train_loss_n9993_s*
  commit "E2, E3: Claude with examples in the prompt and Claude Haiku" \
         results/next/preds_claude_E* results/next/claude_calls_next.json
  commit "Baselines report, error analysis, summary; T1 report with three full-data seeds" \
         results/t1_baselines.md results/next/summary.md results/next/joint_errors_for_review.csv results/t1_report.md
  git status --short
  GIT_TERMINAL_PROMPT=0 git push -u origin next-experiments
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
  cpu_stage scan report -- scan
  cpu_stage commit_push scan -- commit_push
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
