import faiss
import numpy as np

# Load embeddings
embeddings = np.load('doc_embeddings.npy')
print("Embedding shape:", embeddings.shape)
d = embeddings.shape[1]  # should be 384

# FAISS expects float32
if embeddings.dtype != np.float32:
    embeddings = embeddings.astype('float32')

# Build a FAISS index for cosine similarity
index = faiss.IndexFlatIP(d)  # inner product = cosine similarity (since normalized)
index.add(embeddings)

# Save index
faiss.write_index(index, 'faiss_index.idx')
print("FAISS index built and saved. Total vectors:", index.ntotal)
