from file_handler import vector_store, embedding_manager

query_embed = embedding_manager.genrate_embedding(["Harsh Thakur skills experience"])[0]

result = vector_store.collection.query(
    query_embeddings=[query_embed.tolist()],
    n_results=5
)

print("Documents found:", len(result['documents'][0]) if result['documents'] else 0)
print("Distances:", result['distances'][0] if result['distances'] else None)
print("Sample content:", result['documents'][0][0][:200] if result['documents'] and result['documents'][0] else "NONE")