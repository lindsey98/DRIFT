"""``CaMeLDefense``: run the **real** CaMeL pipeline (``PrivilegedLLM``) over an ASB agent's tools.

Rather than re-implement a thinner CaMeL, this drives the actual ``PrivilegedLLM`` element, so the
defense is faithful to the AgentDojo CaMeL and its utility is not understated:

- the up-to-``max_attempts`` code-gen **retry loop with interpreter-error feedback**,
- the **quarantined LLM** (``query_ai_assistant``) for parsing untrusted observations,
- the real ``default_system_prompt_generator`` (documents the sublanguage + advertises every tool),
- untrusted output tagging via CaMeL's own ``AgentDojoFunction`` (tool outputs get
  ``sources.Tool(name)`` with empty inner-sources ⇒ ``is_trusted`` == False).

We only supply the ASB-specific glue: ASB tools wrapped as AgentDojo ``Function``s (benign ones carry
the OPI injection in their observation), an ``EmptyEnv``, the local LLM / q-LLM pointed at the vLLM
endpoint, and an ``ASBSecurityPolicyEngine`` (default-deny allowlist) so the attacker tool is denied.
The attacker tool is registered (hence advertised, exactly like the ASB baseline) so the comparison
is fair; it is simply never allowed to execute.
"""

import os

import openai
from agentdojo import agent_pipeline
from agentdojo import types as ad_types
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime

from src.camel.interpreter import interpreter
from src.camel.models import _force_single_tool_calls, _make_context_safe
from src.camel.pipeline_elements.privileged_llm import PrivilegedLLM

from .policy import make_asb_policy_engine_cls


def _make_local_llm(model_name: str):
    """Local vLLM (OpenAI-compatible) code-generation LLM, built exactly like models.py's local: path."""
    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("LOCAL_API_KEY", "EMPTY")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    _force_single_tool_calls(client)
    llm = agent_pipeline.OpenAILLM(client, model_name, None)
    llm.name = model_name
    return _make_context_safe(llm)  # context overflow -> task fails gracefully, no crash


def _make_qllm_model(model_name: str):
    """Quarantined-LLM model object pointing at the same local endpoint (mirrors models.py)."""
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("LOCAL_API_KEY", "EMPTY")
    return OpenAIModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


def _make_tool_callable(tool_name: str, tool, attack_suffix: str, inject: bool):
    """Expose one ASB tool (parameterless ``run(params=None) -> str``) as a plain callable that
    AgentDojo can introspect. Benign tools carry the OPI injection in their observation."""

    def _fn() -> str:
        out = tool.run(None)
        if inject and attack_suffix:
            out = f"{out}; {attack_suffix}"
        return out

    _fn.__name__ = tool_name
    _fn.__qualname__ = tool_name
    _fn.__doc__ = (getattr(tool, "description", None)
                   or getattr(tool, "tool_description", None) or tool_name)
    return _fn


def _to_asb_messages(ad_messages) -> list[dict]:
    """Flatten AgentDojo ChatMessages to ASB's {role, content(str)} shape for ASR/RR scoring."""
    out = []
    for m in ad_messages:
        content = m.get("content")
        text = ad_types.get_text_content_as_str(content) if content else ""
        out.append({"role": m.get("role", "assistant"), "content": text})
    return out


class CaMeLDefense:
    def __init__(self, args, logger=None):
        self.args = args
        self.model_name = args.llm_name
        self.logger = logger

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.log(msg + "\n", level="info")

    def run(self, task_input: str, tool_list: dict, attacker_tool_name: str,
            attack_suffix: str, tool_descriptions: dict) -> dict:
        """Execute the real CaMeL pipeline for one ASB task.

        Args:
            task_input: the trusted user task (never includes injected text).
            tool_list: {name: ASB tool instance}, INCLUDING the attacker tool.
            attacker_tool_name: name of the injected attacker tool (denied by policy).
            attack_suffix: the OPI attack string appended to benign observations.
            tool_descriptions: {name: description} (unused; taken from the tool objects).
        Returns a dict with a ``messages`` trajectory for ASB's ASR/RR scoring.
        """
        benign_names = [n for n in tool_list if n != attacker_tool_name]

        # Register all tools (benign + attacker) so they are advertised exactly like the ASB
        # baseline; benign ones carry the untrusted OPI injection in their observation.
        runtime = FunctionsRuntime([])
        for name, tool in tool_list.items():
            runtime.register_function(
                _make_tool_callable(name, tool, attack_suffix, inject=(name in benign_names))
            )

        privileged = PrivilegedLLM(
            llm=_make_local_llm(self.model_name),
            security_policy_engine=make_asb_policy_engine_cls(benign_names),
            quarantined_llm_model=_make_qllm_model(self.model_name),
            eval_mode=interpreter.MetadataEvalMode.NORMAL,
        )

        try:
            _, _, _, ad_messages, _ = privileged.query(task_input, runtime, env=EmptyEnv(), messages=[])
        except Exception as e:
            # PrivilegedLLM already handles token-limit / context overflow internally; this is a
            # last-resort guard so any other failure fails the task (utility False) without crashing.
            self._log(f"[CaMeL] pipeline error: {e}")
            msg = [{"role": "user", "content": task_input},
                   {"role": "assistant", "content": f"[CaMeL error]: {e}"}]
            return {"result": msg[-1], "messages": msg, "camel_denied": True}

        messages = _to_asb_messages(ad_messages)
        self._log(f"[CaMeL] produced {len(messages)} messages")
        return {"result": messages[-1] if messages else "", "messages": messages, "camel_denied": False}
