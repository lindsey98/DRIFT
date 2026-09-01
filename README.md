# DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents

## Overview

Official implementation of the NeurIPS 2025 paper "[DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](https://www.arxiv.org/pdf/2506.12104)". DRIFT defends tool-using LLM agents against indirect prompt injection via three components, toggled by CLI flags:

- `--build_constraints` — build an initial function-trajectory plan + parameter checklist from the user query.
- `--injection_isolation` — detect and isolate injected instructions in tool results.
- `--dynamic_validation` — validate each step against the plan/checklist at runtime.

Enabling all three = the **DRIFT defense**; enabling none = the **original model** (undefended baseline).

## Installation

```bash
conda create -n drift python=3.11 && conda activate drift
```

Then install **one** benchmark backend. Both expose the same `agentdojo` package/CLI, so the DRIFT commands below are identical either way — AgentDyn just registers three extra suites.

**AgentDojo** — 4 suites (`banking, slack, travel, workspace`):

```bash
pip install "agentdojo==0.1.35" -r requirements.txt
```

**AgentDyn** — the same 4 suites **plus** `shopping, github, dailylife`. AgentDyn is a drop-in `agentdojo` (same import name, same version `0.1.35`) published only on GitHub, so install it from source **first**; the subsequent `requirements.txt` then sees `agentdojo==0.1.35` already satisfied and leaves the source install in place:

```bash
git clone git@github.com:SaFo-Lab/AgentDyn.git
pip install -e ./AgentDyn
pip install -r requirements.txt
```

Verify which backend is active (the suite list tells you):

```bash
python -c "from agentdojo.task_suite.load_suites import get_suites; print(sorted(get_suites('v1.2')))"
# AgentDojo -> ['banking', 'slack', 'travel', 'workspace']
# AgentDyn  -> ['banking', 'dailylife', 'github', 'shopping', 'slack', 'travel', 'workspace']
```

## Models & API Keys

The provider is selected automatically from the `--model` name. Export the matching key.

| Provider   | `--model` example                              | Env var(s) |
|------------|------------------------------------------------|------------|
| OpenAI     | `gpt-4o-mini-2024-07-18`                        | `OPENAI_API_KEY` |
| Google     | `gemini-2.5-pro`                                | `GOOGLE_API_KEY` (Vertex: `GCP_PROJECT`, `GCP_LOCATION`) |
| Anthropic  | `anthropic:claude-sonnet-4-5-20250929`          | `ANTHROPIC_API_KEY` |
| Local      | `Qwen3-30B-A3B-Instruct-2507`, `Llama-3.3-70B-Instruct`, or `local:<name>` | `LOCAL_API_BASE`, `LOCAL_API_KEY` |
| OpenRouter | anything else (e.g. `meta-llama/Llama-3-70b-chat-hf`) | `OPENROUTER_API_KEY` |

**Local models:** any OpenAI-compatible server (vLLM/SGLang/Ollama) works. 
Bare `qwen…`/`llama…` names (no `/`) and `local:…` route to `LOCAL_API_BASE` (default `http://localhost:8000/v1`, key `EMPTY`). Example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --served-model-name Qwen3-30B-A3B-Instruct-2507 --port 8000
```

## How to Run

All benchmarks use the same entry point:

```bash
python pipeline_main.py --model <MODEL> [ATTACK] [DEFENSE] --suites <SUITES>
```

Pick the `[ATTACK]` and `[DEFENSE]` fragments for the configuration you want:

| Configuration                         | `[ATTACK]`                                          | `[DEFENSE]` |
|---------------------------------------|-----------------------------------------------------|-------------|
| Attack + DRIFT defense                | `--do_attack --attack_type important_instructions`  | `--build_constraints --injection_isolation --dynamic_validation` |
| No attack + DRIFT defense             | *(none)*                                            | `--build_constraints --injection_isolation --dynamic_validation` |
| Attack + original model               | `--do_attack --attack_type important_instructions`  | *(none)* |
| No attack + original model            | *(none)*                                            | *(none)* |

Example (attack + DRIFT defense on AgentDojo):

```bash
python pipeline_main.py \
  --model anthropic:claude-sonnet-4-5-20250929 \
  --do_attack --attack_type important_instructions \
  --build_constraints --injection_isolation --dynamic_validation \
  --suites banking,slack,travel,workspace
```

**Suites** — AgentDojo: `banking,slack,travel,workspace`. AgentDyn adds `shopping,github,dailylife` (install AgentDyn per [Installation](#installation)). Same interface for both; pass any of them to `--suites`, e.g. `--suites shopping,github,dailylife`. Requesting an AgentDyn suite without AgentDyn installed exits with a message telling you to install it.

**Other flags:** `--adaptive_attack`, `--attack_type <name>` (see `utils.py` for the full list), `--target_user_tasks 1,4,7`, `--target_injection_tasks 1,2,3`, `--force_rerun`.

### ChatInject

[ChatInject](https://github.com/hwanchang00/ChatInject) injects the **target model's own chat-template delimiters** into a tool result, closing the current turn and opening a fake system/user/assistant exchange — the role confusion is the exploit. Registered as `--attack_type <name>`:

| `--attack_type` | Template | Payload |
|---|---|---|
| `chat_inject_qwen3` | Qwen3 | single fake system+user turn |
| `chat_inject_glm` | GLM | single fake system+user turn |
| `chat_inject_qwen3_with_utility_system_multiturn_7` | Qwen3 | 7-turn fake dialogue |
| `chat_inject_glm_with_utility_system_multiturn_7` | GLM | 7-turn fake dialogue |
| `chat_inject_qwen3_with_utility_authority_endorsement_system_multiturn_7` | Qwen3 | 7-turn dialogue, authority-endorsement persuasion |
| `chat_inject_glm_with_utility_authority_endorsement_system_multiturn_7` | GLM | 7-turn dialogue, authority-endorsement persuasion |

Pick the variant matching your `--model` (Qwen3 template vs GLM template) — the delimiters are model-specific, so a mismatched template silently no-ops. Example:

```bash
python pipeline_main.py \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --do_attack --attack_type chat_inject_qwen3_with_utility_system_multiturn_7 \
  --build_constraints --injection_isolation --dynamic_validation \
  --suites banking,slack,travel
```

**Coverage:** the multi-turn variants load pre-generated dialogues from `chatinject_data/`, keyed by exact injection-GOAL string, and ChatInject only generated them for **banking / slack / travel**. An uncovered GOAL (e.g. any workspace/shopping/github/dailylife task, or a GOAL AgentDyn reworded) raises a clear `ValueError`. The single-turn `chat_inject_qwen3` / `chat_inject_glm` variants have no data dependency and work on any suite. Template delimiters are defined in `chatinject_attack.py:MODEL_CONFIGS` — verify them against your served model's `tokenizer_config` before trusting the numbers.

### ASB

See [`ASB_DRIFT/README.md`](ASB_DRIFT/README.md).

## Results

Results are written under `logs/<model>[+drift]/<suite>/<user_task>/<attack>/<injection_task>.json` (AgentDojo-style). The `+drift` suffix marks defended runs; the bare model name marks the original model, so the two never collide:

```
logs/Llama-3.3-70B-Instruct+drift/banking/user_task_0/
├── none/none.json                                # benign run
└── important_instructions/injection_task_0.json  # under-attack run
```

Each JSON records the config (defense flags, attack type), full conversation, token usage, and `utility` / `security` outcomes.

## Citation

```
@article{DRIFT,
  title={DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents},
  author={Hao Li and Xiaogeng Liu and Hung-Chun Chiu and Dianqi Li and Ning Zhang and Chaowei Xiao},
  journal={NeurIPS},
  year={2025}
}
```
