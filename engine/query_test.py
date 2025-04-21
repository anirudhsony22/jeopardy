from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# Load FAISS index and doc_id mapping
index = faiss.read_index('faiss_index.idx')
with open('doc_ids.pkl', 'rb') as f:
    doc_ids = pickle.load(f)

# Load title and content mappings
with open('doc_titles.pkl', 'rb') as f:
    doc_titles = pickle.load(f)
with open('doc_contents.pkl', 'rb') as f:
    doc_contents = pickle.load(f)

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Query loop ===
while True:
    query = input("\n🟡 Enter your question (or type 'exit' to quit):\n> ")
    if query.lower().strip() == 'exit':
        break

    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Search for top-1 match
    D, I = index.search(query_embedding, k=1)
    top_idx = I[0][0]
    doc_id = doc_ids[top_idx]

    # Print result
    print("\n🧠 Top Match:")
    print("📄 Title:", doc_titles.get(doc_id, "Unknown Title"))
    print("📝 Snippet:\n", doc_contents.get(doc_id, "[Content not found]")[:500], "...\n")