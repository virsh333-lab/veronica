import os
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.document_loaders import Py
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
import uuid
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from typing import Any, List, Dict

def processall(file_path):
    try:
        loader = PyPDFLoader(file_path)
        document = loader.load()

        for doc in document:
            doc.metadata['source_file'] = Path(file_path).name
            doc.metadata['file_type'] = 'pdf'

        print(f"document loaded successfully, total {len(document)} pages")
        return document
    except Exception as e:
        print(f"here is the error: {e}")
        return []


def process_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        doc = Document(
            page_content=text,
            metadata={"source_file": Path(file_path).name, "file_type": "excel"}
        )
        print(f"excel loaded successfully, {len(df)} rows")
        return [doc]
    except Exception as e:
        print(f"here is the error: {e}")
        return []

def process_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        loader = WebBaseLoader(url, header_template=headers)
        documents = loader.load()
        
        # Basic check - agar content bahut chhota hai ya "access denied" jaisa lagta hai
        for doc in documents:
            if len(doc.page_content) < 200 or "access denied" in doc.page_content.lower():
                print(f"Warning: content looks invalid/blocked for {url}")
                return []
        
        for doc in documents:
            doc.metadata['source_file'] = url
            doc.metadata['file_type'] = 'url'
        print(f"url loaded successfully, {len(documents)} sections")
        return documents
    except Exception as e:
        print(f"here is the error: {e}")
        return []

def process_any(file_path=None, url=None):
    if url:
        return process_url(url)
    elif file_path:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return processall(file_path)   # tera existing PDF function
        elif ext in [".xlsx", ".xls"]:
            return process_excel(file_path)
        else:
            print(f"unsupported file type: {ext}")
            return []
    else:
        return []


# split karo and embed

def split(document, chunk_size=1000, chunk_overlap=200):
    textsplit = RecursiveCharacterTextSplitter(
        chunk_overlap = chunk_overlap,
        chunk_size = chunk_size,
        length_function = len,
        separators=["\n\n","\n",""," "]
    )

    splitdoc = textsplit.split_documents(document)
    print(f"splited succesfully : splited {len(document)} documents and create {len(splitdoc)} chunks")
    if splitdoc:
        print(f"preview  --> {splitdoc[0].page_content[:200]}")
    return splitdoc

class Embedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def genrate_embedding(self, document:list[Any])->np.ndarray:
        if not self.model:
            raise ValueError("model not loaded")
        embedding = self.model.encode(document)
        return embedding


class Vectorstore:
    def __init__(self, collection_name:str="user_document", persist_directory:str="./data/vectorstore"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        try:
            self.client=chromadb.PersistentClient(path= self.persist_directory)
            os.makedirs(self.persist_directory, exist_ok=True)
            # Hard-disk storage setup (Laptop restart par bhi data save rahega)
            self.client = chromadb.PersistentClient(path = self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"user document for rag","hnsw:space": "cosine"}
            )
            print(f"successfully intilaize and create collection table")
        except Exception as e:
            print(f"error is here : {e}")

    def add_document(self, document:list[Any], embeddings:np.ndarray, thread_id: str):
        if len(document) != len(embeddings):
            raise ValueError("len should be equal")

        ids=[]
        metadatas=[]
        documents_list=[]
        embeddings_list=[]

        for i, (doc, embedding) in enumerate(zip(document, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['index']=i
            metadata["content_length"] = len(doc.page_content)
            metadata["thread_id"] = thread_id
            metadatas.append(metadata)

            documents_list.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids = ids,
                embeddings = embeddings_list,
                metadatas= metadatas,
                documents = documents_list
            )
            print(f"succesfully added {len(document)} documents to vector store")
            print(f"total documents in collecton : {self.collection.count()}")
        except Exception as e:
            print(f"error adding in docuemnts :{e}")





# rag retriver

class rag:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: Vectorstore, embedding_manager: Embedding):
        """Initialize the retriever
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embedding
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str,thread_id: str, top_k: int = 3, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for query
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity_score threshold
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")
        
        # 1. Generate query embedding
        query_embedding = self.embedding_manager.genrate_embedding([query])[0]

        # 2. Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where={"thread_id": thread_id}
            )
            
            retrieved_docs = []

            # Check if ChromaDB returned any documents
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                # Loop through all returned matches
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    print("Raw distances:", distances)
                    # Apply score threshold filtering
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                # Loop ke BAHAR print karo ki kitne filtering pass huye
                print(f"Retrieved {len(retrieved_docs)} documents after filtering.")
            else:
                print("No documents found in ChromaDB.")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []



    
