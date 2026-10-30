import os
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient

client = MongoClient(os.getenv("MONGODB_URL"))
db = client["veronica_db"]
chats_collection = db["chats"]
# chats_collection.insert_one({"test": "hello mongo"})
# print("Inserted!")

def save_messages(thread_id:str, role:str, content:str):
    chats_collection.insert_one({
        "thread_id":thread_id,
        "role":role,
        "content":content
    })

def get_history(thread_id:str):
    messages = chats_collection.find({"thread_id":thread_id}).sort("_id", 1)
    history=[]
    for msg in messages:
        history.append({"role":msg['role'], "content":msg['content']})
    return history

def get_all_threads():
    threads_ids = chats_collection.distinct("thread_id")
    threads=[]
    for id in threads_ids:
        first_msg = chats_collection.find_one({"thread_id":id, "role":"user"})
        title = first_msg["content"] if first_msg else "newchat"
        threads.append({"thread_id":id, "title":title})
    return threads

def delete_id(thread_id:str):
    result = chats_collection.delete_many({"thread_id":thread_id})
    return {"deleted_count": result.deleted_count}
