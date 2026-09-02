#!/bin/bash
# Clean (no-attack) run: measures the benign-task utility ceiling. No OPI injection and no attacker
# tool; each task runs once. Traces land at:
#   logs/<model>_nodefense/<agent>/<user_task>/none/none.json          (no defense)
#   logs/<model>+<defense>/<agent>/<user_task>/none/none.json          (with a defense)
#
# Usage (run from the asb/ directory):
#   bash scripts/run_clean.sh          # no defense  -> utility ceiling
#   bash scripts/run_clean.sh camel    # defense on benign tasks -> over-defense baseline
#
# Needs a local OpenAI-compatible endpoint; set LOCAL_BASE_URL / LOCAL_API_KEY if not the defaults
# (http://localhost:8000/v1, "EMPTY").
set -euo pipefail

DEFENSE="${1:-}"                                   # "" or a defense name (e.g. camel, drift)
MODEL="${MODEL:-Qwen3.6-35B-A3B}"                  # must match the served model name
# The attacker-tool file is only used to enumerate which agents/tasks exist; nothing is injected.
ATTACKER_TOOLS="${ATTACKER_TOOLS:-data/all_attack_tools_non_aggressive.jsonl}"
TASK_NUM="${TASK_NUM:-6}"                          # per-agent task cap (>= max tasks -> all tasks)
WORKFLOW_MODE="${WORKFLOW_MODE:-react}"
REACT_MAX_TURNS="${REACT_MAX_TURNS:-6}"
MAX_WORKERS="${MAX_WORKERS:-16}"                   # concurrent tasks (high values exhaust FDs)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
FORCE_RERUN="${FORCE_RERUN:-}"                     # set to 1 to recompute cached tasks

# The react agent doesn't need reasoning; thinking burns the token budget. Disable by default.
export LOCAL_DISABLE_THINKING="${LOCAL_DISABLE_THINKING-1}"

# Local endpoint is on localhost: never route it through an HTTP proxy.
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

# Refuse judge through the same (agent) model by default; override with ASB_JUDGE_MODEL /
# ASB_JUDGE_BASE_URL / ASB_JUDGE_API_KEY (see run_opi.sh header).
export ASB_JUDGE_MODEL="${ASB_JUDGE_MODEL-$MODEL}"

# ASB runs with cwd=asb (aios/, pyopenagi/, camel_adapter/, data/ resolve). Repo root on PYTHONPATH
# so the CaMeL kernel (src.camel) imports for the CaMeL defense.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$(pwd):${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Top trace dir: <model>_nodefense or <model>+<defense>.
if [ -n "$DEFENSE" ]; then _PIPE="${MODEL}+${DEFENSE}"; else _PIPE="${MODEL}_nodefense"; fi
LOG_DIR="${LOG_DIR:-logs/$_PIPE}"

LABEL="clean"; [ -n "$DEFENSE" ] && LABEL="clean_${DEFENSE}"
mkdir -p logs
RES="logs/${MODEL}_${LABEL}.csv"

DEFENSE_ARG=()
[ -n "$DEFENSE" ] && DEFENSE_ARG=(--defense_type "$DEFENSE")
[ -n "$FORCE_RERUN" ] && DEFENSE_ARG+=(--force_rerun)

echo "Model=$MODEL  run=clean${DEFENSE:+ +$DEFENSE}  log_dir=$LOG_DIR"
python main_attacker.py \
  --llm_name "$MODEL" \
  --use_backend local \
  --clean \
  --attacker_tools_path "$ATTACKER_TOOLS" \
  --task_num "$TASK_NUM" \
  --workflow_mode "$WORKFLOW_MODE" \
  --react_max_turns "$REACT_MAX_TURNS" \
  --max_workers "$MAX_WORKERS" \
  --log_dir "$LOG_DIR" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --database /nonexistent_no_memory_db \
  --res_file "$RES" \
  "${DEFENSE_ARG[@]}"
echo "Done. Results CSV: $RES   Traces: $LOG_DIR"
