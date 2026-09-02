import os
import re
import time
import json

from openai import OpenAI
import openai

from .base_llm import BaseLLM
from pyopenagi.utils.chat_template import Response


class LocalLLM(BaseLLM):
    """OpenAI-compatible backend for a locally served model (e.g. vLLM).

    Mirrors ``GPTLLM`` but points the OpenAI client at ``LOCAL_BASE_URL`` /
    ``LOCAL_API_KEY`` (the same env vars the CaMeL repo's ``src/camel/models.py``
    uses for its ``local:`` prefix) and drops the ``gpt`` model-name assertion, so
    any served model name (e.g. ``Qwen3-30B-A3B-Instruct-2507``) works.

    NOTE: if the local endpoint is on localhost and the environment sets
    HTTP(S)_PROXY, make sure NO_PROXY/no_proxy includes ``localhost,127.0.0.1``.
    """

    def __init__(self, llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 1024,
                 log_mode: str = "console"):
        super().__init__(llm_name,
                         max_gpu_memory,
                         eval_device,
                         max_new_tokens,
                         log_mode)

    def load_llm_and_tokenizer(self) -> None:
        base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("LOCAL_API_KEY", "EMPTY")
        self.model = OpenAI(base_url=base_url, api_key=api_key)
        self.tokenizer = None

    def parse_tool_calls(self, tool_calls):
        if tool_calls:
            parsed_tool_calls = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                parsed_tool_calls.append(
                    {
                        "name": function_name,
                        "parameters": function_args,
                    }
                )
            return parsed_tool_calls
        return None

    @staticmethod
    def _coalesce_system_messages(messages):
        """Merge every ``system`` message into a single one at the front.

        ASB appends two system messages (agent description + plan instructions); GPT tolerates
        that, but stricter local chat templates (e.g. Qwen) reject it with
        ``400 "System message must be at the beginning."``. Concatenating them into one leading
        system message keeps the same content while satisfying the template.
        """
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        if len(system_parts) <= 1:
            return messages
        others = [m for m in messages if m.get("role") != "system"]
        merged = {"role": "system", "content": "\n\n".join(p for p in system_parts if p)}
        return [merged, *others]

    @staticmethod
    def _sanitize_tools(tools):
        """Give every tool a valid JSON-schema ``parameters`` object.

        ASB's tool schemas set ``parameters: null`` (and the attacker tool omits the key). OpenAI
        tolerates that, but stricter tool-call parsers (e.g. SGLang ``qwen3_coder``) may fail to
        emit a tool call for such tools, so the model answers in plain text instead. Coerce
        ``parameters`` into ``{"type":"object","properties":{}}`` when it is missing / not a dict.
        """
        if not tools:
            return tools
        fixed = []
        for t in tools:
            t = dict(t)
            fn = dict(t.get("function", {}))
            if not isinstance(fn.get("parameters"), dict):
                fn["parameters"] = {"type": "object", "properties": {}}
            t["function"] = fn
            fixed.append(t)
        return fixed

    def process(self, agent_process, temperature=0.0):
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        messages = self._coalesce_system_messages(agent_process.query.messages)
        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level="executing",
        )
        try:
            kwargs = dict(
                model=self.model_name,
                messages=messages,
                tools=self._sanitize_tools(agent_process.query.tools),
                max_tokens=self.max_new_tokens,
                seed=0,
                temperature=temperature,
            )
            # Single-tool-call parsers (e.g. vLLM llama3_json) reject parallel calls. Some vLLM
            # builds reject the `parallel_tool_calls` param itself with a 400 -> set
            # LOCAL_NO_PARALLEL_TOOL_CALLS_FLAG=1 to skip sending it.
            if agent_process.query.tools and not os.getenv("LOCAL_NO_PARALLEL_TOOL_CALLS_FLAG"):
                kwargs["parallel_tool_calls"] = False
            # Reasoning models (Qwen3 served with --reasoning-parser) burn the token budget on
            # <think> before emitting the JSON plan / tool call, which truncates them. The ASB react
            # agent does not need reasoning, so allow disabling it via LOCAL_DISABLE_THINKING=1
            # (Qwen3 chat template honours chat_template_kwargs.enable_thinking=false).
            if os.getenv("LOCAL_DISABLE_THINKING"):
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            response = self.model.chat.completions.create(**kwargs)
            response_message = response.choices[0].message.content
            tool_calls = self.parse_tool_calls(
                response.choices[0].message.tool_calls
            )
            agent_process.set_response(
                Response(
                    response_message=response_message,
                    tool_calls=tool_calls,
                )
            )
        except openai.APIConnectionError as e:
            self._report_llm_error(agent_process, "Server connection error", e, getattr(e, "__cause__", None))
        except openai.APIStatusError as e:
            # Covers RateLimitError / BadRequestError (subclasses). Surface the real body.
            body = getattr(getattr(e, "response", None), "text", None) or getattr(e, "message", None)
            self._report_llm_error(agent_process, f"STATUS error {getattr(e, 'status_code', '?')}", e, body)
        except Exception as e:
            self._report_llm_error(agent_process, "An unexpected error occurred", e, None)

        agent_process.set_status("done")
        agent_process.set_end_time(time.time())

    def _report_llm_error(self, agent_process, label, exc, detail=None):
        msg = f"{label}: {detail if detail is not None else exc}"
        # Print + log so the real cause (e.g. model-name mismatch, unsupported param) is visible
        # instead of an opaque "STATUS error 400".
        print(f"[LocalLLM] {msg}")
        try:
            self.logger.log(f"[LocalLLM] {msg}\n", level="warning")
        except Exception:
            pass
        agent_process.set_response(Response(response_message=msg))
