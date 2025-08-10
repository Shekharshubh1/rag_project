from fastapi import FastAPI, UploadFile
from utils.query import query_rag
from utils.ingest import ingest_pdf

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    ingest_pdf(file_path)
    return {"message": "PDF processed and added to Pinecone index."}

@app.get("/query")
def query_endpoint(q: str):
    answer = query_rag(q)
    return {"answer": answer}
