import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class QuestionNormalizer:
    def __init__(self, model=None, client=None, api_key=None, base_url=None):
        self.model_name = model or os.getenv("OPENAI_MODEL", "deepseek-chat")
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.getenv("API_KEY"),
                base_url=base_url or os.getenv("API_BASE")
            )

    def normalize_question(self, question: str) -> str:
        prompt = (
            "Chuyển câu hỏi đời thường về giao thông thành một câu hỏi pháp luật rõ ràng, ngắn gọn, "
            "cụ thể hóa chủ thể, hành vi và tình huống nếu cần, dùng đúng ngôn ngữ hỏi trong luật giao thông. "
            "Chỉ trả lại đúng 1 câu hỏi đã được chuẩn hóa, không thêm bất kỳ nhãn, chú thích, phân tích hoặc giải thích nào, không thay đổi các từ ngữ quan trọng, ngắn nhất có thể"
            "không có từ 'Input', 'Output' trong kết quả.\n"
            "Ví dụ:\n"
            "Ô tô vượt đèn đỏ thì làm sao\n"
            "-> Người điều khiển xe ô tô không chấp hành đèn tín hiệu giao thông sẽ bị xử phạt thế nào\n"
            "Tôi lái xe máy trên cao tốc có làm sao không\n"
            "-> Người điều khiển xe mô tô lưu thông trên đường cao tốc bị phạt thế nào?\n"
            f"{question}\n"
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý pháp luật, chuyên chuẩn hóa câu hỏi sang ngôn ngữ pháp luật giao thông."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            result = completion.choices[0].message.content.strip()
            return result
        except Exception as e:
            print("LLM error:", e)
            return question

if __name__ == "__main__":
    normalizer1 = QuestionNormalizer()
    print(normalizer1.normalize_question("Tôi kẹp 3 khi lái xe máy có làm sao không"))