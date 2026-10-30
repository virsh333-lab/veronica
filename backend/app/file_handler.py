from rag import processall, split, Embedding, Vectorstore, rag, process_any
from langchain_core.tools import tool

embedding_manager = Embedding()
vector_store = Vectorstore()
retriever = rag(vector_store, embedding_manager)


# def upload_file(file_path: str):
#     documents = processall(file_path)

#     chunks = split(documents)

#     embeddings = embedding_manager.genrate_embedding(
#         [c.page_content for c in chunks]
#     )

#     vector_store.add_document(chunks, embeddings)

#     return {
#         "status": "success",
#         "chunks_added": len(chunks)
#     }

from rag import process_any, split, Embedding, Vectorstore, rag
from dataclasses import dataclass
from langchain.tools import ToolRuntime
from context import VeronicaContext

def upload_file(file_path: str = None, url: str = None, thread_id: str = None):
    documents = process_any(file_path=file_path, url=url)
    if not documents:
        return {"status": "failed", "message": "Could not extract any content"}
    
    chunks = split(documents)
    embeddings = embedding_manager.genrate_embedding([c.page_content for c in chunks])
    vector_store.add_document(chunks, embeddings, thread_id)
    return {"status": "success", "chunks_added": len(chunks)}

@tool
def file_search(query: str , runtime: ToolRuntime[VeronicaContext]) -> str:
        """Search uploaded files belonging to the current conversation."""
        thread_id = runtime.context.thread_id

        results = retriever.retrieve(query,thread_id=thread_id, top_k=3)

        print("RESULTS:", results)

        if not results:
            return "No relevant information found in the uploaded document."

        combined = "\n\n".join(
            [r["content"] for r in results]
        )

        print("PDF CONTEXT:")
        print(combined)

        return combined


# if __name__ == "__main__":
#     print("Uploading PDF...")

#     result = upload_file()

#     print(result)

#     print("\nSearching PDF...")

#     answer = file_search.invoke("from where harsh did his education")

#     print("\nFINAL RESULT:")
#     print(answer)