"""
OpenAI API backend for LangPert.

Supports OpenAI, Azure OpenAI, and any OpenAI-compatible APIs.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from openai import OpenAI

from .base import BaseBackend


class OpenAIBackend(BaseBackend):
    """Backend for OpenAI and OpenAI-compatible APIs (Azure OpenAI, etc.)."""

    def __init__(self, api_key: str, base_url: Optional[str] = None,
                 model: str = "gpt-4", temperature: float = 0.2,
                 timeout: int = 240, max_retries: int = 20, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, model=model,
                         temperature=temperature, timeout=timeout,
                         max_retries=max_retries, **kwargs)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        self.model = model
        self.temperature = temperature

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None,
                     temperature: Optional[float] = None, verbose: bool = False, **kwargs) -> str:
        """Generate text using OpenAI-compatible API."""
        messages = self._build_messages(prompt, system_prompt)

        if verbose:
            print(f"\nFinal Formatted Messages (OpenAI API):")
            print(f"{'-'*40}")
            for msg in messages:
                print(f"{msg['role'].upper()}: {msg['content']}")
                print(f"{'-'*20}")
            print(f"{'-'*40}")

        kwargs.pop('verbose', None)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            **kwargs
        )

        return completion.choices[0].message.content

    def generate_batch(self, prompts: List[str], max_workers: int = 32, **kwargs) -> List[str]:
        """Generate text for multiple prompts concurrently using threads.

        Args:
            prompts: List of prompts.
            max_workers: Number of concurrent threads (default 32).
            **kwargs: Passed to generate_text.

        Returns:
            List of responses in the same order as prompts.
        """
        results = [None] * len(prompts)

        def _call(idx, prompt):
            return idx, self.generate_text(prompt, **kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_call, i, p): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                idx, response = future.result()
                results[idx] = response

        return results