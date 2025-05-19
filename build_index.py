import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def build_faiss_index(corpus_path="./data/corpus.csv", index_path="./faiss/index.faiss", model_name='VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'):
    print("🔄 Đang tải dữ liệu corpus...")
    df = pd.read_csv(corpus_path)
    texts = df["text"].tolist()
    cids = df["cid"].tolist()

    print(f"🔄 Đang nhúng {len(texts)} đoạn văn...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("🔄 Đang xây FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("💾 Lưu FAISS index và mapping cid...")
    faiss.write_index(index, index_path)
    np.save(index_path.replace(".faiss", "_cid.npy"), np.array(cids))
    np.save(index_path.replace(".faiss", "_text.npy"), np.array(texts))

    print("✅ Xây dựng FAISS index hoàn tất.")

if __name__ == "__main__":
    os.makedirs("./faiss/", exist_ok=True)
    build_faiss_index()
