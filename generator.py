import os
from dotenv import load_dotenv
from retriever import LegalRetriever
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE")  
)

model_name = os.getenv("OPENAI_MODEL", "deepseek-chat")

class LegalRAG_Generator:
    def __init__(self, index_path="./faiss/index.faiss"):
        print("🔄 Đang khởi tạo retriever và client DeepSeek...")
        self.retriever = LegalRetriever(index_path=index_path)

    def generate_answer(self, question, top_k=3):
        docs = self.retriever.retrieve(question, top_k=top_k)
        context = "\n\n".join(list({doc['text'] for doc in docs}))[:2000]
        print(context)

        prompt = (
            f"Dưới đây là một số trích đoạn từ văn bản pháp luật Việt Nam:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            f"Hãy trả lời ngắn gọn và chính xác theo đúng nội dung trên."
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

# if __name__ == "__main__":
#     rag = LegalRAG_Generator()
#     question = "Mức phạt khi không đội mũ bảo hiểm là bao nhiêu?"
#     print("❓", question)
#     print("📝", rag.generate_answer(question))
