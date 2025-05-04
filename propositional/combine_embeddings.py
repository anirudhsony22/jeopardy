import os
import numpy as np
import pickle
import faiss

base_dir = '../granular_bge/'

# === Config
index_file = os.path.join(base_dir, 'bge_index_ivfpq.faiss')
metadata_out_path = os.path.join(base_dir, 'all_metadata_streamed.pkl')

# === FAISS Index Setup (IVFPQ for memory efficiency)
d = 768  # vector dimension
nlist = 128
m = 8
nbits = 8

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# === First pass: train index on a subset
trained = False
for fname in sorted(os.listdir(base_dir)):
    if fname.startswith('embeddings_chunk_') and fname.endswith('.npy'):
        emb_path = os.path.join(base_dir, fname)
        embs = np.load(emb_path)
        if not trained:
            train_samples = embs[:min(len(embs), 10000)]
            index.train(train_samples)
            trained = True
        break  # Only train once

# === Stream through chunks and build index + save metadata
metadata_all = []
for i in sorted([
    int(f.split('_')[-1].split('.')[0])
    for f in os.listdir(base_dir)
    if f.startswith('embeddings_chunk_') and f.endswith('.npy')
]):
    emb_path = os.path.join(base_dir, f'embeddings_chunk_{i}.npy')
    meta_path = os.path.join(base_dir, f'metadata_chunk_{i}.pkl')

    embeddings = np.load(emb_path)
    index.add(embeddings)  # add in chunks

    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
        metadata_all.extend(metadata)

    print(f"✅ Added chunk {i} to index")

# === Save FAISS index and metadata
faiss.write_index(index, index_file)
print(f"✅ Saved FAISS index to {index_file}")

with open(metadata_out_path, 'wb') as f:
    pickle.dump(metadata_all, f)
print(f"✅ Saved all metadata to {metadata_out_path}")
