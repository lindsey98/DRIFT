# This file just wraps around the LLM classes in aios/llm_kernel/llm_classes
# and provides an easy interface for the rest of the code to access
# All abstractions will be implemented here

from .llm_classes.model_registry import MODEL_REGISTRY

# Heavy backends (HfNativeLLM -> torch/transformers, OllamaLLM -> ollama, vLLM) are imported
# lazily inside __init__ so that importing LLMKernel for the OpenAI-compatible / local path does
# not require those packages to be installed.

class LLMKernel:
    def __init__(self,
                 llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 256,
                 log_mode: str = "console",
                 use_backend: str = None
        ):

        # For API-based LLM
        if llm_name in MODEL_REGISTRY.keys():
            self.model = MODEL_REGISTRY[llm_name](
                llm_name = llm_name,
                log_mode = log_mode
            )
        # For locally-deployed LLM
        else:
            if use_backend in ("local", "openai_compatible"):
                # OpenAI-compatible endpoint (e.g. vLLM served with --served-model-name),
                # reached via LOCAL_BASE_URL / LOCAL_API_KEY. No torch/transformers needed.
                from .llm_classes.local_llm import LocalLLM
                self.model = LocalLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode
                )
            elif use_backend == "ollama" or llm_name.startswith("ollama"):
                from .llm_classes.ollama_llm import OllamaLLM
                self.model = OllamaLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode
                )

            elif use_backend == "vllm":
                from .llm_classes.vllm import vLLM
                self.model = vLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode
                )
            else: # use huggingface LLM without backend
                from .llm_classes.hf_native_llm import HfNativeLLM
                self.model = HfNativeLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode
                )

    def address_request(self,
                        agent_process,
                        temperature=0.0):
        self.model.address_request(agent_process,temperature)

    def address_request_list(self,
                        agent_process,
                        temperature=0.0):
        self.model.address_request_list(agent_process,temperature)
