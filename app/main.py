from fastapi import FastAPI, UploadFile
import shutil

from app.rag import process_pdf
from app.vectorstore import create_vectorstore
from app.ask import ask_question

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Research Copilot Running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile):

    pdf_path = f"data/{file.filename}"

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = process_pdf(pdf_path)

    create_vectorstore(chunks)

    return {"message": "PDF uploaded successfully"}

@app.get("/ask")
def ask(query: str):

    answer = ask_question(query)

    return {
        "question": query,
        "answer": answer
    }