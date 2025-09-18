# tools/rag.py
from __future__ import annotations
import re
from typing import List, Tuple, Optional
import numpy as np
from langchain_core.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

def _normalize(text: str) -> str:
    t = text or ""
    t = t.lower().replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()

class HybridRAG:
    def __init__(self, docs: List[str]):
        self.raw_docs = docs
        self.docs = [_normalize(d) for d in docs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
        self.X_tfidf = self.vectorizer.fit_transform(self.docs)
        self.embedder = None
        if HAS_ST:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self.X_emb = self.embedder.encode(self.docs, normalize_embeddings=True)
            except Exception as e:
                print(f"[RAG] Embedding modeli yüklenemedi ({e}), sadece TF-IDF kullanılacak.")
                self.embedder = None

    def query(self, q: str, top_k: int = 5) -> List[Tuple[float, str]]:
        tf_scores = cosine_similarity(self.vectorizer.transform([_normalize(q)]), self.X_tfidf).ravel()
        if self.embedder:
            qv_emb = self.embedder.encode([_normalize(q)], normalize_embeddings=True)[0]
            emb_scores = np.dot(self.X_emb, qv_emb)
            scores = 0.5 * emb_scores + 0.5 * tf_scores
        else:
            scores = tf_scores
        order = np.argsort(-scores)
        return [(float(scores[i]), self.raw_docs[i]) for i in order[:top_k] if scores[i] > 0.1]

_RAG_INSTANCE: Optional[HybridRAG] = None

def initialize_rag(docs: List[str]):
    global _RAG_INSTANCE
    _RAG_INSTANCE = HybridRAG(docs)
    print(f"[RAG] RAG sistemi başarıyla başlatıldı ({'Hybrid' if _RAG_INSTANCE.embedder else 'TF-IDF only'} mod).")

@tool
def schema_search(query: str, top_k: int = 3) -> str:
    """
    Veritabanı şeması içinde ilgili tabloları, sütunları veya metrikleri arar.
    SQL yazmadan önce şemayı anlamak için kullanın. Örneğin, 'toplam gelir' sorusu için 'gelir' diye aratın.
    """
    if _RAG_INSTANCE is None:
        return "Hata: RAG sistemi başlatılmamış."
    results = _RAG_INSTANCE.query(q=query, top_k=top_k)
    if not results:
        return f"'{query}' sorgusu için ilgili şema bilgisi bulunamadı."
    formatted_results = "Potansiyel olarak ilgili bulunan şema parçaları:\n" + "\n".join(
        [f"- {doc} (skor: {score:.2f})" for score, doc in results]
    )
    return formatted_results.strip()