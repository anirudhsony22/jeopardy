from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
import pickle
import os
import argparse

# === Configuration ===
input_file = '../questions.txt'
input_base = "../database/"
chunk_size = 'inf'
faiss_file = "faiss_index.idx"
top_k = 10

# === Models ===
retrieval_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  # Embed queries
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # Rerank top-k


def load_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) % 3 == 0, "❌ Expected triplets of CATEGORY, QUESTION, ANSWER"
    return [(lines[i], lines[i+1], lines[i+2]) for i in range(0, len(lines), 3)]


def load_database(base, chunk_size, faiss_file):
    index = faiss.read_index(os.path.join(base, str(chunk_size), 'faiss', faiss_file))
    with open(os.path.join(base, str(chunk_size), 'comb_chunks', 'doc_ids.pkl'), 'rb') as f:
        doc_ids = pickle.load(f)
    with open(os.path.join(base, str(chunk_size), 'doc_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    with open(os.path.join(base, str(chunk_size), 'doc_titles.pkl'), 'rb') as f:
        titles = pickle.load(f)
    with open(os.path.join(base, str(chunk_size), 'doc_contents.pkl'), 'rb') as f:
        contents = pickle.load(f)
    return index, doc_ids, metadata, titles, contents


def retrieve_top_k(query, index, top_k=10):
    embedding = retrieval_model.encode([query], normalize_embeddings=True)
    D, I = index.search(embedding, k=top_k)
    return I[0]


def rerank_results(query, doc_ids, doc_contents):
    pairs = [(query, doc_contents[i]) for i in doc_ids if i in doc_contents]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(doc_ids, scores), key=lambda x: -x[1])
    return reranked


def evaluate(questions, index, metadata, titles, contents, top_k=10):
    total = len(questions)
    reciprocal_ranks = []

    for category, question, answer in questions:
        query = f"{category} : {question}"
        initial_hits = retrieve_top_k(query, index, top_k)
        reranked_hits = rerank_results(query, initial_hits, contents)

        found_rank = None

        for rank, (doc_id, _) in enumerate(reranked_hits, start=1):
            original_doc_id = metadata.get(doc_id, {}).get('original_doc_id', None)
            if original_doc_id is None:
                continue
            content = contents.get(original_doc_id, "").lower()
            if answer.lower() in content:
                found_rank = rank
                break

        if found_rank:
            reciprocal_ranks.append(1 / found_rank)
            result = f"✅ (Rank {found_rank})"
        else:
            reciprocal_ranks.append(0.0)
            result = "❌"

        print(f"{result} [{category}] Q: {question}")
        print(f"    ➤ Expected: {answer}")

    mrr = sum(reciprocal_ranks) / total
    print(f"\n📊 MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print(f"    Total Questions: {total}")
    print(f"    Correct@1: {sum(1 for r in reciprocal_ranks if r == 1.0)}")
    return mrr, total

# === Main Execution ===
if __name__ == "__main__":
    questions = load_lines(input_file)
    index, doc_ids, metadata, titles, contents = load_database(input_base, chunk_size, faiss_file)
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', default=10)
    correct, total = evaluate(questions, index, metadata, titles, contents, top_k=10)

    print(f"\n📊 Accuracy: {correct}/{total} = {100 * correct / total:.2f}%")