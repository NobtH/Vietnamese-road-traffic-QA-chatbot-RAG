# preprocess_dataset.py
import pandas as pd
from datasets import Dataset

def preprocess_and_save(csv_path="./data/Dataset_6000.csv", output_path="./vit5_dataset"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["context", "question", "answer"])

    # Tạo input/output cho T5
    df["input"] = "Câu hỏi: " + df["question"] + "\nVăn bản: " + df["context"]
    df["output"] = df["answer"]

    # Chuyển thành HuggingFace Dataset
    dataset = Dataset.from_pandas(df[["input", "output"]])
    dataset = dataset.train_test_split(test_size=0.1)
    dataset.save_to_disk(output_path)
    print("✅ Đã lưu dataset tại:", output_path)

if __name__ == "__main__":
    preprocess_and_save()
