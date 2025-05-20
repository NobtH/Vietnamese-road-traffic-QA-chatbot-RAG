# reranker.py
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', top_k: int = 3):
        print(f"🔄 Loading reranker model: {model_name}")
        self.reranker = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, question: str, documents: list[dict]) -> list[dict]:
        if not documents:
            return []

        pairs = [[question, doc['text']] for doc in documents]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_k]]

