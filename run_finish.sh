#!/usr/bin/env bash
# Finishing pipeline for T1: full-training-set run, figures, reports, checks, summary, push.
# Safe to re-run: each stage has a .done_<stage> marker in the status dir and finished work is skipped.
#
#   bash run_finish.sh          run every stage in order
#   FINISH_DRY=1 bash run_finish.sh   test run (tiny training, separate folder, no commit or push)
#
# Env: SCAN_SCRIPT = path to a private-detail scanner that supports --json (the scan stage is
# skipped, and so is the push, if it is not set); PUSH=0 to commit without pushing.
set -u
cd "$(dirname "$(readlink -f "$0")")"
VENV=${VENV:-.venv}
[ -f "$VENV/bin/activate" ] || VENV=${MAIN_VENV:-}
# shellcheck disable=SC1090
[ -n "$VENV" ] && source "$VENV/bin/activate"
for v in $(compgen -e | grep -E '^(CLAUDE|AI_AGENT)'); do unset "$v"; done
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DRY=${FINISH_DRY:-0}
D=results/finish; [ "$DRY" = 1 ] && D=results/finish_dryrun/status
mkdir -p "$D/logs"
OOM_WAIT=600; [ "$DRY" = 1 ] && OOM_WAIT=10
GPU_WAIT_MAX=21600; GPU_POLL=300
MODE=finish

say()   { echo "[$(date '+%F %T')] [finish] $*"; }
ts()    { while IFS= read -r l; do printf '%s %s\n' "$(date '+%F %T')" "$l"; done; }
status(){ python stress_common.py status --dir "$D" --pipeline finish >/dev/null 2>&1 || true; }
mark()  { python stress_common.py mark --dir "$D" --stage "$1" --state "$2" --reason "${3:-}" --log "$D/logs/$1.log" --pipeline finish >/dev/null 2>&1 || true; }
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

# cpu_stage NAME [deps...] -- command...
cpu_stage(){
  local st=$1; shift; local deps=()
  while [ "$1" != "--" ]; do deps+=("$1"); shift; done; shift
  if is_done "$st"; then say "$st already done"; return 0; fi
  for dep in "${deps[@]}"; do
    if ! is_done "$dep"; then mark "$st" skipped "dependency $dep not done"; say "$st skipped ($dep not done)"; status; return 0; fi
  done
  mark "$st" running; say "$st starting"
  if run_logged "$st" "$@"; then mark "$st" done; say "$st done"; else mark "$st" failed "exit code $?"; say "$st FAILED"; fi
  status
}

stage_full_train(){
  local st=full_train attempt=1 extra="" gpu rc
  if is_done $st; then say "$st already done"; return 0; fi
  mark $st running
  while :; do
    if ! gpu=$(wait_gpu); then mark $st failed "no GPU memory (no GPU had 11 GB free for 6 h)"; return 0; fi
    say "$st on GPU $gpu (attempt $attempt${extra:+, $extra})"
    CUDA_VISIBLE_DEVICES=$gpu run_logged $st python full_data.py $extra; rc=$?
    case $rc in
      0) mark $st done; say "$st done"; return 0;;
      3) if [ $attempt -eq 1 ]; then say "CUDA OOM; waiting $OOM_WAIT s, then retrying"; sleep $OOM_WAIT; attempt=2
         elif [ $attempt -eq 2 ]; then say "CUDA OOM again; low-memory fallback (batch 2 x accum 4, eval batch 8)"; attempt=3; extra="--low_mem"
         else mark $st failed "CUDA OOM after a retry and the low-memory fallback"; return 0; fi;;
      4) mark $st failed "NaN/inf training loss (learning rate not changed)"; return 0;;
      *) mark $st failed "exit code $rc"; return 0;;
    esac
  done
}

check_light(){
  # report scripts must run without PyTorch and reproduce the reports apart from the timestamp line
  local LV=.venv-light T
  if [ ! -x "$LV/bin/python" ]; then
    python3.11 -m venv "$LV" && "$LV/bin/pip" install -q "datasets==5.2.0" "scikit-learn==1.9.1" || return 1
  fi
  "$LV/bin/python" -c "import importlib.util as u; assert not any(u.find_spec(m) for m in ('torch','peft','transformers')); print('light venv: no torch/peft/transformers')" || return 1
  T=$(mktemp -d)
  cp results/t1_report.md results/t1_robustness.md "$T/"
  "$LV/bin/python" make_t1_report.py && "$LV/bin/python" make_robustness_report.py || { cp "$T"/*.md results/; return 1; }
  local ok=0
  for f in t1_report.md t1_robustness.md; do
    if diff <(grep -v '^Generated' "$T/$f") <(grep -v '^Generated' "results/$f") >/dev/null; then echo "$f: identical apart from the timestamp line"
    else echo "$f: DIFFERS"; diff <(grep -v '^Generated' "$T/$f") <(grep -v '^Generated' "results/$f") | head -20; ok=1; fi
  done
  cp "$T"/*.md results/; rm -rf "$T"
  for f in *.py model_card/*.py; do python -m py_compile "$f" || ok=1; done
  bash -n run_stress.sh && bash -n run_finish.sh || ok=1
  return $ok
}

check_scan(){
  [ -n "${SCAN_SCRIPT:-}" ] && [ -f "$SCAN_SCRIPT" ] || { echo "SCAN_SCRIPT not set; cannot scan"; return 1; }
  python "$SCAN_SCRIPT" . --check
}

commit_push(){
  if [ "$DRY" = 1 ]; then echo "test run: would commit"; git status --short; return 0; fi
  local c=0
  commit(){ local msg=$1; shift; git add -- "$@" 2>/dev/null; if ! git diff --cached --quiet; then git commit -q -m "$msg" && c=$((c+1)) && echo "committed: $msg"; fi; }
  commit "Full-training-set run (9,993 labels, seed 0)" full_data.py results/preds_Qwen3-4B-Instruct-2507_n9993_s0.jsonl \
         results/exp1_Qwen3-4B-Instruct-2507_n9993_s0.json results/train_loss_n9993_s0.json
  commit "Model card: precise definition of the seed-to-seed change; verification from the Hub" \
         make_model_card.py model_card/README.md results/model_card_verification.json
  commit "Figures; T1 report with the full-training-set row, latency and cost section and figures" \
         make_figures.py results/figures make_t1_report.py results/t1_report.md results/t1_robustness.md
  commit "Licence holder, reproduction guide, requirements, finishing pipeline" \
         LICENSE REPRODUCE.md requirements.txt .gitignore run_finish.sh stress_common.py make_finish_summary.py
  commit "Finishing summary" results/finish_summary.md
  echo "$c commit(s); remaining changes:"; git status --short
  [ "$(git branch --show-current)" = finish-t1 ] || { echo "not on finish-t1; refusing to push"; return 1; }
  if [ "${PUSH:-1}" = 1 ]; then GIT_TERMINAL_PROMPT=0 git push -u origin finish-t1 || return 1; fi
}

echo $$ > "$D/.mode_${MODE}_running"
trap 'rm -f "$D/.mode_${MODE}_running"; touch "$D/.mode_${MODE}_exited"; status' EXIT
say "starting (status dir $D, test run=$DRY, pid $$)"
status
stage_full_train; status
cpu_stage full_collect full_train -- python full_data.py --collect --log "$D/logs/full_train.log"
cpu_stage figures -- python make_figures.py
cpu_stage reports -- python make_t1_report.py
cpu_stage verify_light reports -- check_light
cpu_stage scan reports -- check_scan
cpu_stage summary -- python make_finish_summary.py
cpu_stage commit_push scan -- commit_push
say "finished"
