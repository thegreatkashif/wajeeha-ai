from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from config.settings import AppConfig, Secrets


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    raw: object | None = None


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._default_max_tokens = max_tokens

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens or self._default_max_tokens,
            system=system or "",
            messages=messages,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(text=text, provider=self.name, model=self._model, raw=resp)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        import openai

        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._default_max_tokens = max_tokens

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        full_messages = list(messages)
        if system:
            full_messages = [{"role": "system", "content": system}] + full_messages
        resp = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens or self._default_max_tokens,
            messages=full_messages,
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, provider=self.name, model=self._model, raw=resp)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model_name = model
        self._default_max_tokens = max_tokens
        self._genai = genai

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model = self._genai.GenerativeModel(
            self._model_name, system_instruction=system or None
        )
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        resp = await model.generate_content_async(
            prompt,
            generation_config={"max_output_tokens": max_tokens or self._default_max_tokens},
        )
        return LLMResponse(text=resp.text, provider=self.name, model=self._model_name, raw=resp)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        import ollama

        self._client = ollama.AsyncClient(host=base_url)
        self._model = model

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        full_messages = list(messages)
        if system:
            full_messages = [{"role": "system", "content": system}] + full_messages
        resp = await self._client.chat(model=self._model, messages=full_messages)
        return LLMResponse(
            text=resp["message"]["content"], provider=self.name, model=self._model, raw=resp
        )


class LLMRouter:
    """Builds and caches provider instances from config, and routes a
    completion request to a named provider (or the configured default)."""

    def __init__(self, config: AppConfig, secrets: Secrets) -> None:
        self._config = config
        self._secrets = secrets
        self._providers: dict[str, LLMProvider] = {}

    def _build(self, name: str) -> LLMProvider:
        cfg = self._config.llm.providers.get(name)
        if cfg is None:
            raise ValueError(f"No LLM provider configured under llm.providers.{name}")

        if name == "anthropic":
            if not self._secrets.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
            return AnthropicProvider(self._secrets.anthropic_api_key, cfg.model, cfg.max_tokens)
        if name == "openai":
            if not self._secrets.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in .env")
            return OpenAIProvider(self._secrets.openai_api_key, cfg.model, cfg.max_tokens)
        if name == "gemini":
            if not self._secrets.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set in .env")
            return GeminiProvider(self._secrets.gemini_api_key, cfg.model, cfg.max_tokens)
        if name == "ollama":
            return OllamaProvider(cfg.base_url or "http://localhost:11434", cfg.model)

        raise ValueError(f"Unknown provider '{name}'")

    def get(self, name: str | None = None) -> LLMProvider:
        provider_name = name or self._config.llm.default_provider
        if provider_name not in self._providers:
            self._providers[provider_name] = self._build(provider_name)
        return self._providers[provider_name]

    async def complete(
        self,
        messages: list[dict],
        provider: str | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return await self.get(provider).complete(messages, system=system, max_tokens=max_tokens)