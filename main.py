from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import askquery
from db import get_history, get_all_threads, delete_id
from fastapi import Form

class chatRequest(BaseModel):
    query:str
    thread_id:str

app= FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.post("/chat")
# def chat(request: chatRequest):

#     return askquery(
#         request.query,
#         request.thread_id
#     )

@app.post("/chat")
def chat(request:chatRequest):
    return StreamingResponse(askquery(request.query, request.thread_id), media_type="text/plain")

@app.get("/threads")
def threads():
    return get_all_threads()

@app.get("/history/{thread_id}")
def history(thread_id:str):
    return get_history(thread_id)



from fastapi import UploadFile, File
from file_handler import upload_file
import shutil
import os


# @app.post("/upload")
# def upload(file: UploadFile = File(...)):
#     file_path = f"uploaded_pdfs/{file.filename}"
#     os.makedirs("uploaded_pdfs", exist_ok=True)

#     with open(file_path, "wb") as f:
#         f.write(file.file.read())

#     return upload_file(file_path)
@app.post("/upload")
def upload(file: UploadFile = File(None), url: str = Form(None), thread_id: str = Form(...)):
    file_path = None
    
    if file:
        file_path = f"uploaded_pdfs/{file.filename}"
        os.makedirs("uploaded_pdfs", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

    return upload_file(file_path=file_path, url=url, thread_id=thread_id)

@app.delete("/thread/{thread_id}")
def delete(thread_id:str):
    return delete_id(thread_id)