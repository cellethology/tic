# file: tic/annotation/llm_backends/base.py
class BaseLLM:
    """
    Base class for all LLM backends. Defines the interface.
    """
    def generate(self, user_input: str, system_prompt: str = "") -> str:
        raise NotImplementedError("Subclasses must implement the generate method.")