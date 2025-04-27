import faiss
import numpy as np
import pickle
import argparse
import os

def build_faiss_index(input_base = "../database", output_folder = "../database", chunk_size=128, embedding_file='doc_embeddings.npy', output_file='faiss_index.idx', model_name='multi-qa-MiniLM-L6-cos-v1'):
    if chunk_size==0:
        chunk_size = 'inf'
    embedding_folder = os.path.join(input_base, str(chunk_size), model_name, "comb_chunks/")
    # os.makedirs(embedding_folder, exist_ok=True)
    
    print(embedding_folder+embedding_file)
    if not os.path.exists(embedding_folder+embedding_file):
        raise FileNotFoundError(f"Embedding file not found: {embedding_folder+embedding_file}")

    embeddings = np.load(embedding_folder + embedding_file)
    print("Loaded embeddings:", embeddings.shape)

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype('float32')

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    output_folder = os.path.join(output_folder, str(chunk_size), model_name, "faiss/")
    os.makedirs(output_folder, exist_ok=True)
    faiss.write_index(index, output_folder + output_file)
    print(f"FAISS index built and saved to {output_file}")
    print(f"Total vectors indexed: {index.ntotal}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from document embeddings")
    parser.add_argument('--embedding_folder', default='../database/')
    parser.add_argument('--output_folder', default='../database/')
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--model_name', default='multi-qa-MiniLM-L6-cos-v1')
    args = parser.parse_args()

    build_faiss_index(args.embedding_folder, args.output_folder, args.chunk_size, model_name = args.model_name)