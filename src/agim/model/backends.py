"""Model backends for AGI Personal Memory."""
from abc import ABC, abstractmethod


class TextModelBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def source(self) -> str: ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str: ...


class StaticTextModelBackend(TextModelBackend):
    @property
    def name(self) -> str:
        return "static-fallback"

    @property
    def source(self) -> str:
        return "static"

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return "[Static model response]"


class HuggingFaceTextBackend(TextModelBackend):
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return f"hf:{self.model_name}"

    @property
    def source(self) -> str:
        return "hf_model"

    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "cuda:0",
                        **kwargs) -> "HuggingFaceTextBackend":
        return cls(model_name, device=device)

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.bfloat16,
                device_map=self.device, local_files_only=True)
            self._model.eval()

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        import torch
        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
