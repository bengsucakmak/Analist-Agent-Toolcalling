# llm.py — OpenRouter + Hugging Face fallback, exponential backoff, circuit breaker
from __future__ import annotations
import os
import time
import random
import json
import typing as t

# LangChain taban sınıfları
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

# OpenAI-uyumlu istemci (OpenRouter için)
try:
    from langchain_openai import ChatOpenAI  # pip install langchain-openai
except Exception:
    ChatOpenAI = None  # opsiyonel

# Basit yardımcılar
def _read(cfg: dict, path: t.List[str], default=None):
    cur = cfg
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


# ─────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────
class _Circuit:
    """Transient hata eşiği aşıldığında belirli süre sağlayıcıyı devre dışı bırakır."""
    def __init__(self, threshold: int, window_sec: int):
        self.threshold = max(1, int(threshold))
        self.window = max(1, int(window_sec))
        self.err_count = 0
        self.open_until = 0.0

    def record_error(self):
        self.err_count += 1
        if self.err_count >= self.threshold:
            self.open_until = time.time() + self.window
            self.err_count = 0

    def record_success(self):
        self.err_count = 0
        self.open_until = 0.0

    @property
    def is_open(self) -> bool:
        return time.time() < self.open_until


def _is_transient_error(e: Exception) -> bool:
    s = str(e).lower()
    keys = [
        "rate limit", "429", "timeout", "timed out",
        "temporarily", "try again", "overloaded",
        "upstream error", "502", "503", "504"
    ]
    return any(k in s for k in keys)


def _sleep_expo(attempt: int, base: float, cap: float):
    # jitter'lı exponential backoff
    wait = min(cap, base * (2 ** attempt))
    # 0.5x - 1.0x arası jitter
    jitter = 0.5 + random.random() * 0.5
    time.sleep(wait * jitter)


# ─────────────────────────────────────────────
# Fallback Chat wrapper
# ─────────────────────────────────────────────
class _FallbackChat:
    """
    Birden fazla sağlayıcıyı sırayla dener.
    invoke(messages, **kwargs) dışında ek API sağlamaz.
    """
    def __init__(
        self,
        models: t.List[tuple[str, BaseChatModel]],
        max_retries_primary: int = 2,
        backoff_base_sec: float = 0.3,
        backoff_cap_sec: float = 2.0,
        breaker_threshold: int = 3,
        breaker_window_sec: int = 60,
    ):
        if not models:
            raise ValueError("No chat models provided.")
        self.models = models
        self.primary = models[0][0]
        self.breakers = {name: _Circuit(breaker_threshold, breaker_window_sec) for name, _ in models}
        self.max_retries_primary = max(0, int(max_retries_primary))
        self.backoff_base = float(backoff_base_sec)
        self.backoff_cap = float(backoff_cap_sec)

    def invoke(self, messages: t.List[BaseMessage], **kwargs):
        last_err: Exception | None = None

        for idx, (name, model) in enumerate(self.models):
            br = self.breakers[name]
            if br.is_open:
                # devre açık → sağlayıcıyı atla
                continue

            # Birincil için retry, diğerleri tek deneme
            tries = self.max_retries_primary + 1 if idx == 0 else 1

            for att in range(tries):
                try:
                    resp = model.invoke(messages, **kwargs)
                    br.record_success()
                    return resp
                except Exception as e:
                    last_err = e
                    if _is_transient_error(e):
                        br.record_error()
                        if att < tries - 1:  # tekrar dene
                            _sleep_expo(att, self.backoff_base, self.backoff_cap)
                            continue
                        # retry bitti → sıradaki sağlayıcıya geç
                    else:
                        # kalıcı hata → sıradakine geç
                        break

        # Hepsi başarısız
        raise last_err if last_err else RuntimeError("All providers failed.")


# ─────────────────────────────────────────────
# Hugging Face Inference API (serverless) basit sarmalayıcı
# ─────────────────────────────────────────────
class _HFChatShim(BaseChatModel):
    """
    HF Inference API'ye (serverless) basit HTTP çağrısı yapan ChatModel-benzeri sınıf.
    Notlar:
      - Tool calling desteklemez (bind_tools çağrısı no-op).
      - Mesajları düz metin prompt'a dönüştürür (ROLE prefix'leri ile).
      - Ücretsiz ama oran sınırlı kullanım için uygundur.
    """
    def __init__(self, model_id: str, token: str, temperature: float = 0.1, max_new_tokens: int = 512, timeout: int = 60):
        self.model_id = model_id
        self.token = token
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.timeout = int(timeout)
        self.api = f"https://api-inference.huggingface.co/models/{model_id}"

    # LangChain BaseChatModel uyumu
    def bind_tools(self, tools: t.List) -> "BaseChatModel":  # tools yok sayılır
        return self

    def _messages_to_prompt(self, messages: t.List[BaseMessage]) -> str:
        lines: t.List[str] = []
        for m in messages:
            role = getattr(m, "type", "human")
            content = getattr(m, "content", "")
            if role == "system":
                lines.append(f"[SYSTEM] {content}")
            elif role in ("human", "user"):
                lines.append(f"[USER] {content}")
            else:
                lines.append(f"[ASSISTANT] {content}")
        return "\n".join(lines).strip()

    def invoke(self, messages: t.List[BaseMessage], **kwargs) -> AIMessage:
        import requests
        prompt = self._messages_to_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                # "return_full_text": False,  # bazı modellerde desteklenir
            }
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        r = requests.post(self.api, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Tipik dönüşler:
        #   [{"generated_text": "..."}]
        #   veya {"error": "..."}
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return AIMessage(content=data[0]["generated_text"])
        # Bazı modeller farklı şema döndürebilir → ham JSON string
        return AIMessage(content=json.dumps(data, ensure_ascii=False))


# ─────────────────────────────────────────────
# LLMService
# ─────────────────────────────────────────────
class LLMService:
    """
    Kullanım:
        llm_svc = LLMService(
            provider_priority=["openrouter","hf"],
            provider_configs=cfg["llm"],   # config.yaml'daki llm bloğu
            tools=[...],                   # LangChain tool listesi
            temperature=0.1,
        )
        agent_llm = llm_svc.llm   # .invoke(messages) destekler

    Config beklenen alanlar (örnek):
      llm:
        provider_priority: ["openrouter","hf"]
        temperature: 0.1
        openrouter:
          base_url: "https://openrouter.ai/api/v1"
          model_name: "meta-llama/llama-3.2-3b-instruct:free"
          # api_key: "..."  # yoksa OPENROUTER_API_KEY ortam değişkeni
        hf:
          model_id: "meta-llama/Meta-Llama-3-8B-Instruct"
          # token: "..."    # yoksa HUGGINGFACE_API_TOKEN ortam değişkeni
        fallback:
          max_retries_primary: 2
          backoff_base_sec: 0.3
          backoff_cap_sec: 2.0
          breaker_error_threshold: 3
          breaker_window_sec: 60
    """
    def __init__(
        self,
        provider: str | None = None,           # eski imza ile uyum için (opsiyonel)
        model_name: str | None = None,         # eski imza ile uyum için (opsiyonel)
        temperature: float = 0.1,
        api_key: str | None = None,            # eski imza ile uyum için (opsiyonel)
        tools: t.Optional[t.List] = None,
        provider_priority: t.Optional[t.List[str]] = None,
        provider_configs: t.Optional[dict] = None,
    ):
        self._tools = tools or []
        self._temperature = float(temperature)
        self._cfg = provider_configs or {}

        # Sağlayıcı sırası
        self._priority: t.List[str] = provider_priority or [provider or "openrouter"]

        # Fallback ayarları
        fb = self._cfg.get("fallback", {}) if isinstance(self._cfg, dict) else {}
        self._fb_max_retries = int(_read(fb, ["max_retries_primary"], 2))
        self._fb_backoff_base = float(_read(fb, ["backoff_base_sec"], 0.3))
        self._fb_backoff_cap = float(_read(fb, ["backoff_cap_sec"], 2.0))
        self._fb_breaker_thr = int(_read(fb, ["breaker_error_threshold"], 3))
        self._fb_breaker_win = int(_read(fb, ["breaker_window_sec"], 60))

        # Sağlayıcıları sırayla hazırla
        self._models: t.List[tuple[str, BaseChatModel]] = []
        for name in self._priority:
            m = self._build_provider_model(
                name=name,
                default_provider=provider,
                default_model=model_name,
                default_api_key=api_key,
            )
            if m is None:
                continue
            bound = m.bind_tools(self._tools) if self._tools and hasattr(m, "bind_tools") else m
            self._models.append((name, bound))

        if not self._models:
            raise RuntimeError("LLMService: no usable providers configured.")

        # Dışa verilen model: fallback wrapper
        self.llm = _FallbackChat(
            self._models,
            max_retries_primary=self._fb_max_retries,
            backoff_base_sec=self._fb_backoff_base,
            backoff_cap_sec=self._fb_backoff_cap,
            breaker_threshold=self._fb_breaker_thr,
            breaker_window_sec=self._fb_breaker_win,
        )

    # Sağlayıcı kurulumları
    def _build_provider_model(
        self,
        name: str,
        default_provider: str | None = None,
        default_model: str | None = None,
        default_api_key: str | None = None,
    ) -> BaseChatModel | None:
        name = (name or "").lower()

        # ── OpenRouter (OpenAI-uyumlu) ───────────────────────────
        if name == "openrouter":
            if ChatOpenAI is None:
                return None
            sec = self._cfg.get("openrouter", {}) if isinstance(self._cfg, dict) else {}
            base_url = _read(sec, ["base_url"], "https://openrouter.ai/api/v1")
            api_key = _read(sec, ["api_key"], os.getenv("OPENROUTER_API_KEY") or default_api_key)
            model_name = _read(sec, ["model_name"], default_model or "meta-llama/llama-3.2-3b-instruct:free")
            if not api_key:
                return None  # anahtar yoksa bu sağlayıcıyı atla
            return ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                temperature=self._temperature,
                default_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Analist AI Agent"
                },
            )


        # ── Hugging Face Inference API (serverless) ──────────────
        if name in ("hf", "huggingface", "hugging_face"):
            sec = self._cfg.get("hf", {}) if isinstance(self._cfg, dict) else {}
            model_id = _read(sec, ["model_id"], default_model or "meta-llama/Meta-Llama-3-8B-Instruct")
            token = _read(sec, ["token"], os.getenv("HUGGINGFACE_API_TOKEN"))
            if not token:
                return None
            # Uyarı: tool-calling gerektiren akışlarda HF shim tool çağrısı üretmez.
            return _HFChatShim(model_id=model_id, token=token, temperature=self._temperature)

        # bilinmeyen sağlayıcı
        return None
