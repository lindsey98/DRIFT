# DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents

[Hao Li](https://leolee99.github.io/), [Xiaogeng Liu](https://sheltonliu-n.github.io/), [Hung-Chun Chiu](https://qhjchc.notion.site/), [Dianqi Li](https://scholar.google.com/citations?user=K40nbiQAAAAJ&hl=en), [Ning Zhang](https://cybersecurity.seas.wustl.edu/index.html), [Chaowei Xiao](https://xiaocw11.github.io/).

<a href='https://safo-lab.github.io/DRIFT'><img src='https://img.shields.io/badge/Website-Page-%23f08080'></a> 
<a href='https://github.com/SaFo-Lab/DRIFT'><img src='https://img.shields.io/badge/GitHub-code-green?logo=github'></a> 
<a href='https://arxiv.org/pdf/2506.12104'><img src='https://img.shields.io/badge/Paper-PDF-red?logo=open-access'></a> 
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="Github license"/></a>

<p align="center" width="80%">
<a target="_blank"><img src="assets/framework.png" alt="framework" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

The official implementation of NeurIPS 2025 paper "[DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](https://www.arxiv.org/pdf/2506.12104)".

## Update
- [2026.4.19] 🛠️ Update the evaluation on AgentDyn.
- [2026.1.30] 🛠️ Support the evaluation on more agents.
- [2026.1.30] 🛠️ Update the evaluation code on ASB.


## Overview

DRIFT is an indirect prompt injection (IPI) defense for tool-using LLM agents. It
guards the agent with three components, all driven from `pipeline_main.py`:

- `--build_constraints` — build an initial function-trajectory plan and parameter checklist from the user query.
- `--injection_isolation` — detect and isolate injected instructions found inside tool results.
- `--dynamic_validation` — validate each step against the plan/checklist as the agent runs.

Throughout this README we refer to running with **all three flags enabled** as the
**IPI guard defense**, and running with **none of them** as the **original model**
(the undefended baseline). Attacks are toggled with `--do_attack` and selected with
`--attack_type` (e.g., `important_instructions`).


## Installation

```bash
conda create -n drift python=3.11
source activate drift
pip install "agentdojo==0.1.35"
pip install -r requirements.txt
```

## API Keys

The model provider is chosen automatically from the `--model` name:

| Provider   | Model prefix                                  | Environment variable |
|------------|-----------------------------------------------|----------------------|
| OpenAI     | `gpt-...`                                      | `OPENAI_API_KEY`     |
| Google     | `gemini-...`                                   | `GOOGLE_API_KEY` (Vertex: `GCP_PROJECT`, `GCP_LOCATION`) |
| Anthropic  | `anthropic:claude-...` or `claude-...`         | `ANTHROPIC_API_KEY`  |
| Local      | `local:...`, or a bare `qwen...` / `llama...` name (e.g. `Qwen3-30B-A3B-Instruct-2507`, `Llama-3.3-70B-Instruct`) | `LOCAL_API_BASE`, `LOCAL_API_KEY` |
| OpenRouter | anything else                                  | `OPENROUTER_API_KEY` |

Set whichever key matches the model you plan to run:

```bash
export OPENAI_API_KEY=your_key
export GOOGLE_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export OPENROUTER_API_KEY=your_key
```

### Locally-served models (e.g. Qwen3-30B-A3B-Instruct-2507, Llama-3.3-70B-Instruct)

Any OpenAI-compatible server (vLLM, SGLang, Ollama, ...) is supported. A model is
routed to the local endpoint when its name is prefixed with `local:`, or when it is a
bare `qwen...` / `llama...` served-model name (case-insensitive, with no provider
`/` — names like `meta-llama/...` still go to OpenRouter). For example, serve a model
with vLLM:

```bash
# Qwen
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name Qwen3-30B-A3B-Instruct-2507 \
  --port 8000

# Llama
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --served-model-name Llama-3.3-70B-Instruct \
  --port 8000
```

Then point DRIFT at it (defaults: `http://localhost:8000/v1` and key `EMPTY`):

```bash
export LOCAL_API_BASE=http://localhost:8000/v1   # optional, this is the default
export LOCAL_API_KEY=EMPTY                        # optional, this is the default
```

and pass e.g. `--model Qwen3-30B-A3B-Instruct-2507` or `--model Llama-3.3-70B-Instruct`
(or `--model local:<your-served-name>`) to any of the commands below.

## How to Run (AgentDojo)

The examples below use the Anthropic model `anthropic:claude-sonnet-4-5-20250929`.
You can swap in any supported model (e.g. `gpt-4o-mini-2024-07-18`,
`gemini-2.5-pro`) by changing the `--model` flag. The `--suites` flag selects the
AgentDojo suites to evaluate (`banking,slack,travel,workspace`).

### 1. Important Instructions attack + IPI guard defense

The defense (all three flags) running against the `important_instructions` attack:

```bash
python pipeline_main.py \
  --model anthropic:claude-sonnet-4-5-20250929 \
  --do_attack --attack_type important_instructions \
  --build_constraints --injection_isolation --dynamic_validation \
  --suites banking,slack,travel,workspace
```

### 2. No attack + IPI guard defense

The defense running on clean tasks (measures utility under DRIFT, no injection):

```bash
python pipeline_main.py \
  --model anthropic:claude-sonnet-4-5-20250929 \
  --build_constraints --injection_isolation --dynamic_validation \
  --suites banking,slack,travel,workspace
```

### 3. Important Instructions attack + original model

The undefended baseline against the `important_instructions` attack (no DRIFT flags):

```bash
python pipeline_main.py \
  --model anthropic:claude-sonnet-4-5-20250929 \
  --do_attack --attack_type important_instructions \
  --suites banking,slack,travel,workspace
```

### 4. No attack + original model

The undefended baseline on clean tasks (measures vanilla utility):

```bash
python pipeline_main.py \
  --model anthropic:claude-sonnet-4-5-20250929 \
  --suites banking,slack,travel,workspace
```

### Useful options

- `--adaptive_attack` — evaluate under the adaptive attack.
- `--attack_type` — choose the attack, e.g. `direct`, `ignore_previous`, `system_message`, `injecagent`, `tool_knowledge`, `important_instructions`, and several DoS variants (see `utils.py` for the full list).
- `--target_user_tasks 1,4,7` / `--target_injection_tasks 1,2,3` — run a subset of tasks.
- `--force_rerun` — re-run tasks even if a cached result already exists.


## Evaluating on AgentDyn

To evaluate on AgentDyn, replace the AgentDojo dependency with the AgentDyn version. First, run:

```bash
git clone git@github.com:SaFo-Lab/AgentDyn.git
cd AgentDyn
pip install -e .
```

This additionally supports the `shopping`, `github`, and `dailylife` suites. Use the
same commands as above, swapping the suites — for example, the IPI guard defense
under attack:

```bash
python pipeline_main.py \
  --model anthropic:claude-sonnet-4-5-20250929 \
  --do_attack --attack_type important_instructions \
  --build_constraints --injection_isolation --dynamic_validation \
  --suites shopping,github,dailylife
```

And the original model with no attack:

```bash
python pipeline_main.py \
  --model anthropic:claude-sonnet-4-5-20250929 \
  --suites shopping,github,dailylife
```


## Evaluating on ASB
Please refer to `ASB_DRIFT/README.md`.


## Inspect Results
Results are written under `logs/`, following AgentDojo's per-task layout. The model
folder carries a `+drift` suffix when the DRIFT defense is enabled and is the bare
model name for the undefended (original) model, so defended and baseline runs never
collide:

```
logs/
└── Llama-3.3-70B-Instruct+drift/        # bare "Llama-3.3-70B-Instruct" for the original model
    └── banking/
        └── user_task_0/
            ├── none/none.json                               # benign run
            └── important_instructions/injection_task_0.json # under-attack run
```

Each task writes a JSON file recording the configuration (which defense flags were
on, attack type), the full conversation, token usage, and the `utility` / `security`
outcomes.


## References

If you find this work useful in your research or applications, we appreciate that if you can kindly cite:

```
@articles{DRIFT,
  title={DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents},
  author={Hao Li and Xiaogeng Liu and Hung-Chun Chiu and Dianqi Li and Ning Zhang and Chaowei Xiao},
  journal = {NeurIPS},
  year={2025}
}
```
</content>
</invoke>
