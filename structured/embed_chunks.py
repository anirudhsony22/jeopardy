import os
import pickle_utils as pkl
import pickle
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import subprocess

def ensure_chunks_exist(chunk_size, stride, base_dir='../database'):
    if chunk_size==0:
        chunk_size = 'inf'
    chunk_dir = os.path.join(base_dir, f'{chunk_size}')
    content_file = os.path.join(chunk_dir, 'doc_contents.pkl')

    if not os.path.exists(content_file):
        print(f"Chunks not found for chunk size {chunk_size}. Running chunk_documents.py ...")
        subprocess.run([
            'python', 'chunk_documents.py',
            '--chunk_size', str(chunk_size),
            '--stride', str(stride),
            '--input_dir', base_dir,
            '--output_dir', base_dir
        ])
    else:
        print(f"Found chunked documents in {chunk_dir}")
    return chunk_dir

def embed_documents(chunk_dir, chunk_size, model_name='all-MiniLM-L6-v2', output_base='../database/'):
    if chunk_size==0:
        chunk_size = 'inf'
    save_dir = os.path.join(output_base, str(chunk_size), model_name)
    os.makedirs(save_dir, exist_ok=True)

    save_dir = os.path.join(save_dir, "emb_chunks")
    os.makedirs(save_dir, exist_ok=True)

    # Load chunked doc contents
    doc_contents = pkl.open_pickle(os.path.join(chunk_dir, 'doc_contents.pkl'))

    doc_ids = sorted(doc_contents.keys())
    texts = [doc_contents[i] for i in doc_ids]

    model = SentenceTransformer(model_name)

    CHUNK_SIZE = 10000
    total_chunks = (len(texts) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_index in range(total_chunks):
        start = chunk_index * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(texts))
        emb_file = os.path.join(save_dir, f'embeddings_chunk_{chunk_index}.npy')
        id_file = os.path.join(save_dir, f'doc_ids_chunk_{chunk_index}.pkl')

        if os.path.exists(emb_file) and os.path.exists(id_file):
            print(f"Skipping chunk {chunk_index} (already encoded)")
            continue

        print(f"Encoding chunk {chunk_index} — docs {start} to {end - 1}")
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
        pkl.save_pickle(chunk_ids, f'{id_file}')

        print(f"Saved: {emb_file}, {id_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunked documents using BERT")
    parser.add_argument('--chunk_size', type=int, default=128, help='Chunk size used for chunking')
    parser.add_argument('--stride', type=int, default=64, help='Stride used for chunking')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model')
    args = parser.parse_args()

    chunk_dir = ensure_chunks_exist(args.chunk_size, args.stride)
    embed_documents(chunk_dir, args.chunk_size, model_name=args.model)
