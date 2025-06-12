import os
from dotenv import load_dotenv
from retriever import LegalRetriever
from openai import OpenAI
from rerank import Reranker
from normalize_question import QuestionNormalizer

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE")  
)

model_name = os.getenv("OPENAI_MODEL", "deepseek-chat")

class LegalRAG_Generator:
    def __init__(self, index_path="./faiss/index.faiss"):
        print("🔄 Đang khởi tạo retriever...")
        self.retriever = LegalRetriever(index_path=index_path)
        self.reranker = Reranker(top_k=10)
        self.normalizer = QuestionNormalizer(model=model_name, client=client)

    def generate_answer(self, question, top_k=5):
        question = self.normalizer.normalize_question(question)
        print("\n**************Normalizes question*************\n")
        print(question)

        docs = self.retriever.retrieve(question, top_k=10)
        print("\n**************Context Origin***************\n")
        print("\n\n".join(list({doc['text'] for doc in docs})))

        docs = self.reranker.rerank(question, docs)
        print("\n**************Context after rerank******************\n")
        print("\n\n".join(list({doc['text'] for doc in docs})))

        top_docs = docs[:top_k]
        context = "\n\n".join(list({doc['text'] for doc in top_docs}))
        print("\n***********Final context******************\n")
        print(context)

        prompt = (
            f"Dưới đây là một số trích đoạn từ văn bản pháp luật Việt Nam:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            f"Hãy trả lời ngắn gọn và chính xác theo đúng nội dung trên."
            f"Trong câu trả lời thêm cả phần trích dẫn nguồn của câu trả lời."
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý pháp lý tiếng Việt."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )

        return response.choices[0].message.content

if __name__ == "__main__":
    rag = LegalRAG_Generator()
    question = "Tôi uống rượu lái xe máy thì làm sao"
    print("❓", question)
    print("📝", rag.generate_answer(question))
