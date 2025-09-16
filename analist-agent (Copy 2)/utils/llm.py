import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import ChatOpenAI
from utils.cost import CostTracker

# LLM yardımcı modülü için logger
log = logging.getLogger("llm")

class LLMService:
    """
    OpenAI-compatible LLM istemcisi (LangChain ChatOpenAI üzerinden).
    Üst katmanlara basit bir get_text(system, user) arayüzü sağlar.
    """
    def __init__(
        self,
        model_name: str = "llama3.3-70b-q8",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        base_url: str = "http://10.150.96.44:20004/v1",
        api_key: str = "dummy",
        **kwargs
    ):
        # --- Normalizasyon: 'model' ve 'model_name' çakışmalarını önle ---
        # Dışarıdan gelebilecek alternatif anahtarlar kwargs içinde olabilir.
        # ChatOpenAI'ye aynı argümanı iki kez geçmemek için temizliyoruz.
        model_from_kwargs = kwargs.pop("model", None)
        alt_model_name = kwargs.pop("model_name", None)
        model = model_from_kwargs or model_name or alt_model_name or "llama3.3-70b-q8"

        # Konfig parametrelerini sakla
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # LangChain ChatOpenAI istemcisi:
        # - OpenAI uyumlu endpoint'e bağlanır
        # - invoke(messages) ile çağrılır
        self.llm = ChatOpenAI(
            base_url=base_url,
            model=model,                # Tek kaynak: normalize edilmiş model
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            **kwargs                    # Artık 'model'/'model_name' burada yok
        )

    @retry(
        reraise=True,                                 # hata sürerse çağırana fırlat
        stop=stop_after_attempt(3),                   # en fazla 3 deneme
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),  # üstel bekleme
        retry=retry_if_exception_type(Exception),     # Exception tipinde ise yeniden dene
    )
    def get_text(self, system: str, user: str, **kwargs) -> str:
        """
        LangChain ChatOpenAI arayüzü: system + user ile çağır, düz metin döndür.
        Retry dekoratörü kısa süreli hatalarda otomatik tekrar dener.
        """
        messages = [
            {"role":"system","content":system},
            {"role":"user","content":user},
        ]
        # invoke: LangChain'in ChatModel arayüzü; AIMessage döner
        resp = self.llm.invoke(messages, **kwargs)
        # Farklı sunucular farklı alanlar döndürebilir; 'content' öncelik, yoksa str(resp)
        text = getattr(resp, "content", None) or str(resp)
        return text

def call_llm_text(llm: LLMService, system: str, user: str, cost: Optional[CostTracker]=None, **kwargs) -> str:
    """
    Yüksek seviyeli yardımcı:
      - LLMService.get_text'i çağırır
      - CostTracker'a yaklaşık token/maliyet ekler (varsa)
      - Üretilen metni döndürür
    """
    out = llm.get_text(system, user, **kwargs)
    if cost is not None:
        # Yaklaşık token hesabı: gerçek usage metrikleri yoksa prompt+cevap üzerinden
        cost.add_call(system + "\n" + user, out)
    log.debug("LLM çıktı uzunluğu: %d", len(out))
    return out
