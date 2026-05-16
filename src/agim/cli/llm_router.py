"""LLM-based Intent Router — replaces regex with language model classification."""
from ..core.state import Intent


class LLMIntentRouter:
    """Classify user intent using a small language model instead of regex patterns.

    Uses a fine-tuned prompt for structured intent classification with 8 intents.
    Falls back to regex-based IntentRouter if no LLM is available.
    """

    SYSTEM_PROMPT = """Classify the user's intent into exactly one of these categories:

- fact_teach: user is teaching a new fact (declarative statement, not a question)
- fact_correct: user is correcting a previous mistake ("no, actually...", "that's wrong...")
- fact_question: user is asking a question (ends with "?" or starts with "what/who/when/where/why/how")
- preference: user is expressing a preference about how to respond ("I prefer...", "always...", "never...")
- feedback: user is giving feedback on a response ("good answer", "that was wrong")
- forget: user wants to delete/undo something
- history: user wants to see what they've taught
- stats: user wants statistics

Reply with ONLY the intent name, nothing else. Examples:
"Paris is the capital of France" → fact_teach
"What is the capital of France?" → fact_question
"No, actually Napoleon was born in 1769" → fact_correct
"I prefer short answers" → preference
"show me what I taught you" → history
"How many facts do I have?" → stats
"""

    def __init__(self, model=None, tokenizer=None):
        self._model = model
        self._tokenizer = tokenizer
        self._fallback = None

    @property
    def fallback(self):
        if self._fallback is None:
            from .intent_router import IntentRouter
            self._fallback = IntentRouter()
        return self._fallback

    def _llm_classify(self, text: str) -> Intent | None:
        if self._model is None or self._tokenizer is None:
            return None
        try:
            import torch
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id)
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = response.split("→")[-1].strip().lower()
            if "assistant" in result:
                result = result.split("assistant")[-1].strip()
            for intent in Intent:
                if intent.value in result:
                    return intent
            return None
        except Exception:
            return None

    def route(self, text: str) -> Intent:
        llm_result = self._llm_classify(text)
        if llm_result is not None:
            return llm_result
        return self.fallback.route(text)
