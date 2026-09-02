#!/bin/bash
# Run the ASB Observation Prompt Injection (OPI) benchmark on a locally-served model.
#
# Usage:
#   bash scripts/run_opi.sh              # undefended baseline (expect ASR > 0)
#   bash scripts/run_opi.sh camel        # CaMeL defense
#   bash scripts/run_opi.sh drift        # DRIFT defense (trajectory + injection isolation)
#
# Run from the asb/ directory. Requires a vLLM (or other OpenAI-compatible) endpoint; set
# LOCAL_BASE_URL / LOCAL_API_KEY if not the defaults (http://localhost:8000/v1, "EMPTY").
set -euo pipefail

DEFENSE="${1:-}"                                   # "" (baseline), "camel", or "drift"
MODEL="${MODEL:-Qwen3.6-35B-A3B}"      # must match the vLLM --served-model-name
ATTACK_TYPE="${ATTACK_TYPE:-context_ignoring}"
ATTACKER_TOOLS="${ATTACKER_TOOLS:-data/attack_tools_test.jsonl}"  # small slice for a smoke test
TASK_NUM="${TASK_NUM:-5}"
# automatic = model plans itself (use this for real results); manual = use each agent's hardcoded
# plan (diagnostic: removes the planning variable, only affects the baseline, not CaMeL);
# react = dynamic tool-calling loop (full OPI attack surface). Each ASB agent has only 2 normal
# tools, so ~4 turns suffice; REACT_MAX_TURNS lets you raise the cap if a task ever hits it.
WORKFLOW_MODE="${WORKFLOW_MODE:-react}"
REACT_MAX_TURNS="${REACT_MAX_TURNS:-6}"
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

echo "Model=$MODEL  attack_type=$ATTACK_TYPE  defense=${LABEL}  attacker_tools=$ATTACKER_TOOLS"
python main_attacker.py \
  --llm_name "$MODEL" \
  --use_backend local \
  --observation_prompt_injection \
  --attack_type "$ATTACK_TYPE" \
  --attacker_tools_path "$ATTACKER_TOOLS" \
  --task_num "$TASK_NUM" \
  --workflow_mode "$WORKFLOW_MODE" \
  --react_max_turns "$REACT_MAX_TURNS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --database /nonexistent_no_memory_db \
  --res_file "$RES" \
  "${DEFENSE_ARG[@]}"
echo "Done. Results CSV: $RES"
