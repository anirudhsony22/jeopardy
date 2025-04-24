import numpy as np
import pickle
import os
import glob
import argparse

def combine_embedding_chunks(chunk_size, input_base='../database/', output_dir='../database/'):
    if chunk_size==0:
        chunk_size = 'inf'
    chunk_dir = os.path.join(input_base, str(chunk_size), "emb_chunks")

    emb_files = sorted(glob.glob(os.path.join(chunk_dir, 'embeddings_chunk_*.npy')))
    id_files = sorted(glob.glob(os.path.join(chunk_dir, 'doc_ids_chunk_*.pkl')))
    op_path = os.path.join(output_dir, str(chunk_size))
    os.makedirs(op_path, exist_ok=True)
    op_path = os.path.join(op_path, "comb_chunks")
    os.makedirs(op_path, exist_ok=True)

    all_embeds = [np.load(f) for f in emb_files]
    combined_embeddings = np.vstack(all_embeds)

    all_ids = []
    for f in id_files:
        with open(f, 'rb') as pf:
            all_ids.extend(pickle.load(pf))

    # Save merged outputs
    np.save(os.path.join(op_path, 'doc_embeddings.npy'), combined_embeddings)
    with open(os.path.join(op_path, 'doc_ids.pkl'), 'wb') as f:
        pickle.dump(all_ids, f)

    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    print(f"Total doc IDs: {len(all_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--input_base', default='../database/')
    parser.add_argument('--output_dir', default='../database/')
    args = parser.parse_args()

    combine_embedding_chunks(args.chunk_size, input_base=args.input_base, output_dir=args.output_dir)