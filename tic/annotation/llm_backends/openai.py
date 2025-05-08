#file: tic/annotation/llm_backends/openai.py
import os
import requests
from .base import BaseLLM

class OpenAIChatGPT(BaseLLM):
    """
    OpenAI ChatGPT backend with optional proxy mirror support.

    Parameters
    ----------
    model : str
        Model name, e.g., "gpt-4o-mini" or "gpt-3.5-turbo".
    api_key : str, optional
        OpenAI API key. If not provided, read from OPENAI_API_KEY env var.
    mirror : bool, optional (default: False)
        If True, route requests through the mirror proxy at https://api.openai-proxy.org/v1
    """
    def __init__(self, model: str = "gpt-4.1", **kwargs):
        self.model = model
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in env or passed as api_key.")

        self.mirror = kwargs.get("mirror", False)
        if not self.mirror:
            try:
                import openai  # noqa: F811
            except ImportError:
                raise ImportError(
                    "To use the OpenAI backend without mirror you must install the `openai` package: pip install openai"
                )
            # Ensure the official client uses our key
            os.environ["OPENAI_API_KEY"] = self.api_key
        else:
            # Set base URL for mirror proxy
            self.base_url = "https://api.openai-proxy.org/v1"

    def generate(self, user_input: str, system_prompt: str = "") -> str:
        # Build messages sequence
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})

        if self.mirror:
            # Direct HTTP call to proxy endpoint
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {"model": self.model, "messages": messages}
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            import openai  # noqa: F811
            # Use official OpenAI Python client
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages
            )
            return resp.choices[0].message.content.strip()
