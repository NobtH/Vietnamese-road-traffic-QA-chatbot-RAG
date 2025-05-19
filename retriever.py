from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class LegalRetriever:
    def __init__(self, index_path="./faiss/index.faiss", model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"):
        print("🔄 Đang tải mô hình và index...")
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.cids = np.load(index_path.replace(".faiss", "_cid.npy"), allow_pickle=True)
        self.texts = np.load(index_path.replace(".faiss", "_text.npy"), allow_pickle=True)

    def retrieve(self, query, top_k=5):
        print(f"🔍 Truy vấn: {query}")
        query_vec = self.model.encode([query])
        D, I = self.index.search(query_vec, top_k)
        results = []
        for idx in I[0]:
            results.append({
                "cid": self.cids[idx],
                "text": self.texts[idx]
            })
        return results

# Ví dụ sử dụng:
if __name__ == "__main__":
    retriever = LegalRetriever()
    query = "Mức phạt nếu không chấp hành đèn tín hiệu"
    results = retriever.retrieve(query)
    for i, res in enumerate(results):
        print(f"[{i+1}] CID: {res['cid']}\n{res['text']}\n---")
