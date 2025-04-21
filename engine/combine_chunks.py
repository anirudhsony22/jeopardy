import numpy as np
import pickle
import os
import glob

SAVE_DIR = 'emb_chunks/'

# Gather all chunk files
emb_files = sorted(glob.glob(os.path.join(SAVE_DIR, 'embeddings_chunk_*.npy')))
id_files = sorted(glob.glob(os.path.join(SAVE_DIR, 'doc_ids_chunk_*.pkl')))

# Combine embeddings
all_embeds = [np.load(f) for f in emb_files]
combined_embeddings = np.vstack(all_embeds)

# Combine doc IDs
all_ids = []
for f in id_files:
    with open(f, 'rb') as pf:
        all_ids.extend(pickle.load(pf))

# Save combined outputs
np.save('doc_embeddings.npy', combined_embeddings)
with open('doc_ids.pkl', 'wb') as f:
    pickle.dump(all_ids, f)

print("✅ Combined embeddings shape:", combined_embeddings.shape)
print("✅ Total doc IDs:", len(all_ids))