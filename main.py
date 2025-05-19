from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generator import LegalRAG_Generator  
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = LegalRAG_Generator(index_path="./faiss/index.faiss")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: QuestionRequest):
    question = payload.question
    answer = rag.generate_answer(question)
    return {"answer": answer}

@app.get("/files")
def list_files():
    files = [f for f in os.listdir("data") if f.endswith(".docx")]
    return {"files": files}
app.mount("/files", StaticFiles(directory="data"), name="files")


