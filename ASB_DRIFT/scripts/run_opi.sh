#!/bin/bash

set -euo pipefail

DEFENSE="${1:-}"                                   # "" (baseline), "camel", or "drift"
MODEL="${MODEL:-Qwen3.6-35B-A3B}"      # must match the vLLM --served-model-name
ATTACK_TYPE="${ATTACK_TYPE:-context_ignoring}"
ATTACKER_TOOLS="${ATTACKER_TOOLS:-data/all_attack_tools_non_aggressive.jsonl}"  # small slice for a smoke test
TASK_NUM="${TASK_NUM:-6}"

WORKFLOW_MODE="${WORKFLOW_MODE:-react}"
REACT_MAX_TURNS="${REACT_MAX_TURNS:-6}"
# How many tool observations carry the OPI injection. 0 = every one (ASB default, unrealistic);
# 1 = only the first tool result (a single compromised data source, the realistic case).
OPI_INJECT_LIMIT="${OPI_INJECT_LIMIT:-1}"
# Resume by default: tasks whose JSON trace already exists are skipped (reusing their metrics).
# Set FORCE_RERUN=1 to recompute everything from scratch.
FORCE_RERUN="${FORCE_RERUN:-}"
# For a no-attack (clean) utility run, use the dedicated scripts/run_clean.sh instead.
# Concurrent tasks. ASB's original 5000 exhausts file descriptors and overloads the LLM server;
# keep it modest and match it to what your SGLang endpoint can handle.
MAX_WORKERS="${MAX_WORKERS:-16}"
# Where the per-task JSON traces go. Top dir encodes the run: <model>_nodefense or <model>+<defense>,
# e.g. logs/Qwen3.6-35B-A3B_nodefense/... and logs/Qwen3.6-35B-A3B+camel/...
if [ -n "$DEFENSE" ]; then _PIPE="${MODEL}+${DEFENSE}"; else _PIPE="${MODEL}_nodefense"; fi
LOG_DIR="${LOG_DIR:-logs/$_PIPE}"
# Reasoning models (Qwen3 with --reasoning-parser) spend tokens on <think> before the tool call;
# ASB's default 256 truncates them. Bump the generation budget.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
# The ASB react agent doesn't need reasoning; thinking just burns the token budget and truncates
# the JSON plan / tool call. Disable it by default (set LOCAL_DISABLE_THINKING="" to re-enable).
export LOCAL_DISABLE_THINKING="${LOCAL_DISABLE_THINKING-1}"

# Local vLLM is on localhost: never route it through an HTTP proxy.
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

# Refuse judge. Defaults to the same (agent) model/endpoint so no OpenAI is needed. The judge
# endpoint is INDEPENDENT of the agent: to keep the agent on your local vLLM but judge with a
# hosted model, export before calling e.g.:
# export ASB_JUDGE_MODEL=glm-5.2
# export ASB_JUDGE_BASE_URL=https://api.modelarts-maas.com/openai/v1
#   export ASB_JUDGE_API_KEY=<key>
# (agent still uses LOCAL_BASE_URL/LOCAL_API_KEY, i.e. localhost:8000 by default.)
# Set ASB_JUDGE_MODEL="" to force the OpenAI gpt-4o-mini judge instead.
export ASB_JUDGE_MODEL="${ASB_JUDGE_MODEL-$MODEL}"

# ASB runs with cwd=asb (so aios/, pyopenagi/, camel_adapter/, data/ resolve). Also put the repo
# root on PYTHONPATH so the CaMeL kernel (src.camel) imports for the CaMeL defense.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$(pwd):${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

LABEL="${DEFENSE:-baseline}"
mkdir -p logs/observation_prompt_injection
RES="logs/observation_prompt_injection/${MODEL}_${ATTACK_TYPE}_${LABEL}.csv"

DEFENSE_ARG=()
[ -n "$DEFENSE" ] && DEFENSE_ARG=(--defense_type "$DEFENSE")
[ -n "$FORCE_RERUN" ] && DEFENSE_ARG+=(--force_rerun)

echo "Model=$MODEL  attack_type=$ATTACK_TYPE  run=${LABEL}  attacker_tools=$ATTACKER_TOOLS"
python main_attacker.py \
  --llm_name "$MODEL" \
  --use_backend local \
  --observation_prompt_injection \
  --attack_type "$ATTACK_TYPE" \
  --attacker_tools_path "$ATTACKER_TOOLS" \
  --task_num "$TASK_NUM" \
  --workflow_mode "$WORKFLOW_MODE" \
  --react_max_turns "$REACT_MAX_TURNS" \
  --opi_inject_limit "$OPI_INJECT_LIMIT" \
  --max_workers "$MAX_WORKERS" \
  ${LOG_DIR:+--log_dir "$LOG_DIR"} \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --database /nonexistent_no_memory_db \
  --res_file "$RES" \
  "${DEFENSE_ARG[@]}"
echo "Done. Results CSV: $RES"
