import pandas as pd
import numpy as np
import faiss
import os
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class LegalRetriever:
    def __init__(self,
                 corpus_path="./data/corpus.csv",
                 index_path="./faiss/index.faiss",
                 model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                 top_k=10,
                 rrf_k=60,
                 keyword_boost=1.5):
        print("🔄 Khởi tạo HybridRetriever...")
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.keyword_boost = keyword_boost

        # Load corpus
        self.df = pd.read_csv(corpus_path)
        self.texts = np.load(index_path.replace(".faiss", "_text.npy"), allow_pickle=True)
        self.cids = np.load(index_path.replace(".faiss", "_cid.npy"), allow_pickle=True)

        # Load FAISS
        self.faiss_index = faiss.read_index(index_path)

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # BM25 corpus
        self.bm25_corpus = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.bm25_corpus)

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def _rrf_fusion(self, faiss_rank, bm25_rank):
        all_idx = set(faiss_rank) | set(bm25_rank)
        scores = {}
        for idx in all_idx:
            r1 = faiss_rank.index(idx) if idx in faiss_rank else self.top_k
            r2 = bm25_rank.index(idx) if idx in bm25_rank else self.top_k
            scores[idx] = 1 / (self.rrf_k + r1) + 1 / (self.rrf_k + r2)
        return scores

    def retrieve(self, query: str):
        print(f"\n🔍 Truy vấn: {query}")
        query_tokens = self._tokenize(query)

        # FAISS dense retrieval
        query_vec = self.model.encode([query])
        D, I = self.faiss_index.search(np.array(query_vec), self.top_k * 3)
        faiss_rank = list(I[0])

        # BM25 sparse retrieval
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_rank = np.argsort(bm25_scores)[::-1][:self.top_k * 3].tolist()

        # RRF fusion
        rrf_scores = self._rrf_fusion(faiss_rank, bm25_rank)

        # Apply keyword boosting
        boosted_results = []
        for idx, score in rrf_scores.items():
            text = self.texts[idx]
            keyword_hits = sum(1 for kw in query_tokens if kw in text.lower())
            boost = self.keyword_boost * keyword_hits
            boosted_results.append({
                "cid": self.cids[idx],
                "text": text,
                "score": score + boost,
                "keyword_hits": keyword_hits
            })

        # Sort and return top_k
        boosted_results.sort(key=lambda x: x["score"], reverse=True)
        return boosted_results[:self.top_k]
    
# if __name__ == "__main__":
#     retriever = LegalRetriever()
#     query = "Người điều khiển xe ô tô không chấp hành tín hiệu đèn giao thông sẽ bị xử phạt thế nào?"
#     results = retriever.retrieve(query)

#     for i, doc in enumerate(results):
#         print(f"[{i+1}] CID: {doc['cid']} | Score: {doc['score']:.2f} | KW Hits: {doc['keyword_hits']}")
#         print(doc["text"])
#         print("---")
