# registering all proprietary llm models in a constant.
# Each backend is imported defensively so that a missing optional SDK (google-generativeai,
# boto3, anthropic) does not break importing the registry for the OpenAI-compatible / local path.

MODEL_REGISTRY = {}

try:
    from .gpt_llm import GPTLLM
    MODEL_REGISTRY.update({
        'gpt-3.5-turbo': GPTLLM,
        'gpt-4-turbo': GPTLLM,
        'gpt-4o': GPTLLM,
        'gpt-4o-2024-08-06': GPTLLM,
        'gpt-4o-mini': GPTLLM,
    })
except ImportError:
    pass

try:
    from .gemini_llm import GeminiLLM
    MODEL_REGISTRY.update({
        "gemini-1.5-flash": GeminiLLM,
        "gemini-1.5-pro": GeminiLLM,
    })
except ImportError:
    pass

try:
    from .claude_llm import ClaudeLLM
    MODEL_REGISTRY['claude-3-5-sonnet-20240620'] = ClaudeLLM
except ImportError:
    pass

try:
    from .bed_rock import BedrockLLM
    MODEL_REGISTRY['bedrock/anthropic.claude-3-haiku-20240307-v1:0'] = BedrockLLM
except ImportError:
    pass
