# DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents

## Overview

Official implementation of the NeurIPS 2025 paper "[DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](https://www.arxiv.org/pdf/2506.12104)". DRIFT defends tool-using LLM agents against indirect prompt injection via three components, enabled together by `--defense drift`:

- **build_constraints** — build an initial function-trajectory plan + parameter checklist from the user query.
- **injection_isolation** — detect and isolate injected instructions in tool results.
- **dynamic_validation** — validate each step against the plan/checklist at runtime.

`--defense drift` = the **DRIFT defense** (all three); `--defense none` = the **original model** (undefended baseline).

## Installation

`agentdojo` comes from the fork [`lindsey98/agentdojo`](https://github.com/lindsey98/agentdojo)
— a drop-in `agentdojo` (same import name / version `0.1.35`) that bundles the original 4
suites (`banking, slack, travel, workspace`) **plus** `shopping, github, dailylife`. Clone it
and install from the local copy (retriable if your network is flaky; you can also download the
repo zip instead of `git clone`), then install the rest:

```bash
conda create -n drift python=3.11 && conda activate drift
git clone https://github.com/lindsey98/agentdojo.git
pip install -e ./agentdojo
pip install -r requirements.txt
```

`requirements.txt` does **not** pin `agentdojo`, so the last step never re-fetches it over the
network. Verify all 7 suites are registered:

```bash
python -c "from agentdojo.task_suite.load_suites import get_suites; print(sorted(get_suites('v1.2')))"
# -> ['banking', 'dailylife', 'github', 'shopping', 'slack', 'travel', 'workspace']
```

## Models & API Keys

The provider is selected automatically from the positional `MODEL` name. Export the matching key.

| Provider   | `MODEL` example                                | Env var(s) |
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

`MODEL` is positional; `--defense` is `none` (undefended original model) or `drift`
(= build_constraints + injection_isolation + dynamic_validation). Results always go under `logs/`.

**Under attack:**

```bash
python pipeline_main.py MODEL --run-attack --attack <name> \
  --suite banking slack travel workspace --defense drift
```

**No attack (benign utility):**

```bash
python pipeline_main.py MODEL \
  --suite banking slack travel workspace --defense drift
```

Example:

```bash
python pipeline_main.py Qwen3.6-35B-A3B --run-attack --attack important_instructions \
  --suite banking slack travel workspace --defense drift
```

- **`--defense none | drift`** — swap `drift` for `none` to run the undefended original model.
- **`--attack <name>`** — any registered agentdojo attack: `important_instructions`, `tool_knowledge`, `data_only_syntactic`, `data_only_semantic`, `chat_inject_qwen3`, `cascade_*`, … (omit `--run-attack` for the benign setting).
- **Suites** — space-separated; all 7 come from the [`lindsey98/agentdojo`](https://github.com/lindsey98/agentdojo) install: `banking slack travel workspace shopping github dailylife`. Requesting a suite the installed `agentdojo` doesn't register exits with a clear message.
- **Task subset** — `--user-task`/`-ut` and `--injection-task`/`-it` take space-separated ids or numbers, e.g. `-ut 1 4 7 -it 0 1` (default: all).
- **Optional** — `--force_rerun` (recompute cached tasks), `--html` (save a rendered `<name>.html` next to each `<name>.json`).
- **Attack modifiers** (compose with `--run-attack`): `--adaptive_attack`, `--align_claim`, `--close_tag`, `--repeated_instruction [--repeat_n N]`.

### ChatInject

[ChatInject](https://github.com/hwanchang00/ChatInject) injects the **target model's own chat-template delimiters** into a tool result, closing the current turn and opening a fake system/user/assistant exchange — the role confusion is the exploit. Pass as `--attack <name>`:

| `--attack` | Template | Payload |
|---|---|---|
| `chat_inject_qwen3` | Qwen3 | single fake system+user turn |
| `chat_inject_glm` | GLM | single fake system+user turn |
| `chat_inject_qwen3_with_utility_system_multiturn_7` | Qwen3 | 7-turn fake dialogue |
| `chat_inject_glm_with_utility_system_multiturn_7` | GLM | 7-turn fake dialogue |
| `chat_inject_qwen3_with_utility_authority_endorsement_system_multiturn_7` | Qwen3 | 7-turn dialogue, authority-endorsement persuasion |
| `chat_inject_glm_with_utility_authority_endorsement_system_multiturn_7` | GLM | 7-turn dialogue, authority-endorsement persuasion |

Pick the variant matching your `MODEL` (Qwen3 template vs GLM template) — the delimiters are model-specific, so a mismatched template silently no-ops. Example:

```bash
python pipeline_main.py Qwen3-30B-A3B-Instruct-2507 \
  --run-attack --attack chat_inject_qwen3_with_utility_system_multiturn_7 \
  --suite banking slack travel --defense drift
```

**Coverage:** the multi-turn variants load pre-generated dialogues from `chatinject_data/`, keyed by exact injection-GOAL string, and ChatInject only generated them for **banking / slack / travel**. An uncovered GOAL (e.g. any workspace/shopping/github/dailylife task, or a GOAL the fork reworded) raises a clear `ValueError`. The single-turn `chat_inject_qwen3` / `chat_inject_glm` variants have no data dependency and work on any suite. Template delimiters are defined in `chatinject_attack.py:MODEL_CONFIGS` — verify them against your served model's `tokenizer_config` before trusting the numbers.

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
