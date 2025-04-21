from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

CHUNK_SIZE = 10000
SAVE_DIR = 'emb_chunks/'

# Load documents
with open('doc_contents.pkl', 'rb') as f:
    doc_contents = pickle.load(f)

doc_ids = sorted(doc_contents.keys())
texts = [doc_contents[i] for i in doc_ids]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure output folder exists
os.makedirs(SAVE_DIR, exist_ok=True)

total_chunks = (len(texts) + CHUNK_SIZE - 1) // CHUNK_SIZE

for chunk_index in range(total_chunks):
    start = chunk_index * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, len(texts))
    emb_file = os.path.join(SAVE_DIR, f'embeddings_chunk_{chunk_index}.npy')
    id_file = os.path.join(SAVE_DIR, f'doc_ids_chunk_{chunk_index}.pkl')

    # Skip if this chunk already exists
    if os.path.exists(emb_file) and os.path.exists(id_file):
        print(f"⏩ Skipping chunk {chunk_index} — already saved")
        continue

    print(f"▶️ Encoding chunk {chunk_index} — docs {start} to {end - 1}")
    chunk_texts = texts[start:end]
    chunk_ids = doc_ids[start:end]

    embeddings = model.encode(
        chunk_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(emb_file, embeddings)
    with open(id_file, 'wb') as f:
        pickle.dump(chunk_ids, f)

    print(f"Saved chunk {chunk_index}: shape {embeddings.shape}")