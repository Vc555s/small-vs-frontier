#!/usr/bin/env bash
# Full-training-set run (9,993 labels, seed 0), then figures and reports, with a check that the
# report scripts run without PyTorch. Resumable: each finished stage leaves a .done_<stage> marker
# in the status folder, so a re-run skips it.
#
#   bash run_full_data.sh                    run every stage in order
#   FULL_DATA_TEST=1 bash run_full_data.sh   test run (tiny training, separate folder)
set -u
cd "$(dirname "$(readlink -f "$0")")"
VENV=${VENV:-.venv}
[ -f "$VENV/bin/activate" ] || VENV=${MAIN_VENV:-}
# shellcheck disable=SC1090
[ -n "$VENV" ] && source "$VENV/bin/activate"
for v in $(compgen -e | grep -E '^(CLAUDE|AI_AGENT)'); do unset "$v"; done
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DRY=${FULL_DATA_TEST:-0}
D=results/full_data/status; [ "$DRY" = 1 ] && D=results/full_data_testrun/status
mkdir -p "$D/logs"
OOM_WAIT=600; [ "$DRY" = 1 ] && OOM_WAIT=10
GPU_WAIT_MAX=21600; GPU_POLL=300
MODE=full_data

say()   { echo "[$(date '+%F %T')] [full_data] $*"; }
ts()    { while IFS= read -r l; do printf '%s %s\n' "$(date '+%F %T')" "$l"; done; }
status(){ python stress_common.py status --dir "$D" --pipeline full_data >/dev/null 2>&1 || true; }
mark()  { python stress_common.py mark --dir "$D" --stage "$1" --state "$2" --reason "${3:-}" --log "$D/logs/$1.log" --pipeline full_data >/dev/null 2>&1 || true; }
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
    CUDA_VISIBLE_DEVICES=$gpu run_logged $st python finetune_qwen_full.py $extra; rc=$?
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
  cp results/main_report.md results/robustness_report.md "$T/"
  "$LV/bin/python" make_main_report.py && "$LV/bin/python" make_robustness_report.py || { cp "$T"/*.md results/; return 1; }
  local ok=0
  for f in main_report.md robustness_report.md; do
    if diff <(grep -v '^Generated' "$T/$f") <(grep -v '^Generated' "results/$f") >/dev/null; then echo "$f: identical apart from the timestamp line"
    else echo "$f: DIFFERS"; diff <(grep -v '^Generated' "$T/$f") <(grep -v '^Generated' "results/$f") | head -20; ok=1; fi
  done
  cp "$T"/*.md results/; rm -rf "$T"
  for f in *.py model_cards/*/*.py; do python -m py_compile "$f" || ok=1; done
  for f in run_*.sh; do bash -n "$f" || ok=1; done
  return $ok
}

echo $$ > "$D/.mode_${MODE}_running"
trap 'rm -f "$D/.mode_${MODE}_running"; touch "$D/.mode_${MODE}_exited"; status' EXIT
say "starting (status dir $D, test run=$DRY, pid $$)"
status
stage_full_train; status
cpu_stage full_collect full_train -- python finetune_qwen_full.py --collect --log "$D/logs/full_train.log"
cpu_stage figures -- python make_figures.py
cpu_stage reports -- python make_main_report.py
cpu_stage verify_light reports -- check_light
say "finished"
