# tools/rag.py
from __future__ import annotations
from typing import List, Tuple, Optional
import re
import numpy as np

# TF-IDF tabanlı metin vektörizasyonu ve benzerlik
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Opsiyonel: SentenceTransformer (embedding) varsa hybrid mod daha güçlü olur
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

def _pick_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.device = _pick_device()
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            # sentence-transformers yoksa TF-IDF’a düş
            self.model = None
            self.device = "cpu"
            print(f"[RAG] sentence-transformers yüklenemedi ({e}); TF-IDF fallback kullanılacak.")

    def encode(self, texts: list[str]) -> "np.ndarray":
        if self.model:
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        # TF-IDF fallback:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X = vec.fit_transform(texts)
        return X.toarray() / (np.linalg.norm(X.toarray(), axis=1, keepdims=True) + 1e-8)

def _normalize(text: str) -> str:
    """Basit normalizasyon: lower, '_' -> ' ', fazla boşlukları temizle.
    Şema satırları kısa olduğu için ağır bir dil işleme yapmıyoruz."""
    t = text or ""
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


class _TFIDFRAG:
    """Sadece TF-IDF tabanlı basit RAG."""
    def __init__(self, docs: List[str]):
        # Orijinal metinler ve normalize edilmiş haller
        self.raw_docs = docs
        self.docs = [_normalize(d) for d in docs]
        # Stopwords kullanmıyoruz (TR/EN karışık kısa satırlar); 1-2 gram tercih ediliyor
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
        self.X = self.vectorizer.fit_transform(self.docs)  # Doküman matrisini hazırla

    def query(self, q: str, top_k: int = 5, min_score: float = 0.1) -> List[Tuple[float, str]]:
        # Sorguyu TF-IDF uzayına projekte et
        qv = self.vectorizer.transform([_normalize(q)])
        # Kozinüs benzerliği ile skorla
        scores = cosine_similarity(qv, self.X).ravel()
        # En yüksekten düşüğe sırala
        order = np.argsort(-scores)
        out: List[Tuple[float, str]] = []
        # İlk top_k sonucu, eşik üstünde ise döndür
        for i in order[:top_k]:
            s = float(scores[i])
            if s >= min_score:
                out.append((s, self.raw_docs[i]))
        return out


class _EmbeddingRAG:
    """Sadece embedding tabanlı RAG (SentenceTransformer gerekir)."""
    def __init__(self, docs: List[str], model_name: str = "paraphrase-MiniLM-L6-v2"):
        if not HAS_ST:
            raise RuntimeError("sentence_transformers yüklü değil.")
        self.raw_docs = docs
        self.docs = [_normalize(d) for d in docs]
        # Embedding modeli
        self.embedder = SentenceTransformer(model_name)
        # Doküman embedding'leri (normalize edilmiş)
        self.X = self.embedder.encode(self.docs, normalize_embeddings=True)

    def query(self, q: str, top_k: int = 5, min_score: float = 0.1) -> List[Tuple[float, str]]:
        # Sorgu embedding'i (normalize edilmiş)
        qv = self.embedder.encode([_normalize(q)], normalize_embeddings=True)[0]
        # Kozinüs için normalize vektörlerle dot product yeterli
        scores = np.dot(self.X, qv)
        order = np.argsort(-scores)
        out: List[Tuple[float, str]] = []
        for i in order[:top_k]:
            s = float(scores[i])
            if s >= min_score:
                out.append((s, self.raw_docs[i]))
        return out


class HybridRAG:
    """
    Hybrid RAG: TF-IDF (+ opsiyonel embedding).
    Embedding yoksa TF-IDF tek başına çalışır.
    """
    def __init__(self, docs: List[str], embed_model: str = "paraphrase-MiniLM-L6-v2", alpha: float = 0.5):
        # TF-IDF tarafı her zaman aktif
        self.tfidf = _TFIDFRAG(docs)
        self.embed: Optional[_EmbeddingRAG] = None
        self.alpha = float(alpha)
        # SentenceTransformer varsa embedding modunu da hazırla
        if HAS_ST:
            try:
                self.embed = _EmbeddingRAG(docs, model_name=embed_model)
            except Exception:
                # Embedding başaramazsa (model indirilemedi vs.) sessizce TF-IDF'e düş
                self.embed = None

    def query(self, q: str, top_k: int = 5, min_score: float = 0.1) -> List[Tuple[float, str]]:
        # TF-IDF skorları
        qv = self.tfidf.vectorizer.transform([_normalize(q)])
        tf_scores = cosine_similarity(qv, self.tfidf.X).ravel()

        # Embedding skorları (varsa) ve karışım
        if self.embed is not None:
            qv_emb = self.embed.embedder.encode([_normalize(q)], normalize_embeddings=True)[0]
            emb_scores = np.dot(self.embed.X, qv_emb)
            scores = self.alpha * emb_scores + (1.0 - self.alpha) * tf_scores
        else:
            scores = tf_scores

        # Skorları sırala ve top_k + min_score filtresi uygula
        order = np.argsort(-scores)
        out: List[Tuple[float, str]] = []
        for i in order[:top_k]:
            s = float(scores[i])
            if s >= min_score:
                out.append((s, self.tfidf.raw_docs[i]))  # Orijinal metni döndür
        return out


# --- Geriye dönük uyumluluk: graph.py SimpleRAG bekliyor ---
class SimpleRAG(HybridRAG):
    """Compatibility alias: SimpleRAG = HybridRAG"""
    pass


def get_rag(docs: List[str], prefer: str = "hybrid") -> HybridRAG | _TFIDFRAG:
    """
    Basit fabrika:
      - prefer='hybrid' → HybridRAG (embedding varsa hibrit, yoksa TF-IDF'e düşer)
      - prefer='tfidf'  → yalın TF-IDF
    """
    if prefer == "tfidf":
        return _TFIDFRAG(docs)
    return HybridRAG(docs)


__all__ = ["SimpleRAG", "HybridRAG", "get_rag"]