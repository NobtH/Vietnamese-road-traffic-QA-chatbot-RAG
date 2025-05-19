import os
import re
import pandas as pd
from docx import Document

cid_counter = 1

def extract_law_name(file_path):
    base = os.path.basename(file_path)
    name = os.path.splitext(base)[0]
    return name.replace("_", "/")

def read_docx(file_path):
    doc = Document(file_path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip() != ""]

def parse_document(paragraphs, law_number):
    global cid_counter
    data = []

    current_dieu = ""
    current_muc = ""
    current_sub = []

    def flush_subs():
        global cid_counter
        nonlocal current_sub
        for sub in current_sub:
            text = "\n".join([law_number, current_dieu, current_muc, sub]).strip()
            data.append({"cid": str(cid_counter), "text": text})
            cid_counter += 1
        current_sub = []

    for para in paragraphs:
        if re.match(r"^Điều\s+\d+", para):
            flush_subs()
            current_dieu = para
            current_muc = ""
        elif re.match(r"^\d+\.", para):
            flush_subs()
            current_muc = para
        elif re.match(r"^[a-zđgh]\)", para, re.IGNORECASE):
            current_sub.append(para)
        else:
            if current_sub:
                current_sub[-1] += " " + para
            elif current_muc:
                current_muc += " " + para
            else:
                current_dieu += " " + para

    flush_subs()
    return data

def build_all_corpus_from_docx(folder_path="./data/"):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            law_number = extract_law_name(file_path)
            print(f"Đang xử lý: {law_number}")
            paragraphs = read_docx(file_path)
            chunks = parse_document(paragraphs, law_number)
            all_chunks.extend(chunks)

    df = pd.DataFrame(all_chunks)[["cid", "text"]]
    df.to_csv("./data/corpus.csv", index=False, encoding="utf-8-sig")
    print("✅ Xuất file corpus.csv hoàn tất.")

if __name__ == "__main__":
    build_all_corpus_from_docx(folder_path="./data/")
