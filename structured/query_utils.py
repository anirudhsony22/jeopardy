import os
import gc
import pickle
import pickle_utils
import numpy as np
from rank_bm25 import BM25Okapi
import faiss


# Batch embed function - to avoid RAM overrun
# DO NOT RUN THIS WITHOUT GPU (MIGHT TAKE A DAY TO RUN)
def embed_documents(docs, model, batch_size=512):
    all_embeddings = []
    num_batches = (len(docs) + batch_size - 1) // batch_size  # Calculate the number of batches
    print(num_batches)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(docs))
        batch = docs[start_index:end_index]
        embeddings = model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
        all_embeddings.append(embeddings)
    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings

def load_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) % 3 == 0, "❌ Expected triplets of CATEGORY, QUESTION, ANSWER"
    return [(lines[i], lines[i+1], lines[i+2]) for i in range(0, len(lines), 3)]


# Tokenize and save chunks
def tokenize_and_save_chunk(contents_slice, output_prefix, chunk_index):
    tokenized_chunk = []
    for doc_id, text in contents_slice.items():
        tokenized_chunk.append(text.lower().split())
    output_filename = f"{output_prefix}_chunk_{chunk_index}.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(tokenized_chunk, f)
    return output_filename

# Build BM25 from disk
def build_bm25_from_disk(tokenized_chunks_files):
    corpus = []
    for file in tokenized_chunks_files:
        print(f"Loading tokenized chunk: {file}")
        with open(file, 'rb') as f:
            corpus.extend(pickle.load(f))
        os.remove(file)
        gc.collect()
    print("✅ Building BM25 object...")
    bm25 = BM25Okapi(corpus)
    return bm25

import gc

def tokenize_and_build_bm25(doc_ids, doc_contents, chunk_size, qutil):
    temp_files = []
    num_chunks = (len(doc_ids) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(doc_ids))
        chunk_ids = doc_ids[start_index:end_index]
        contents_slice = {doc_id: doc_contents[doc_id] for doc_id in chunk_ids}
        
        output_file = qutil.tokenize_and_save_chunk(contents_slice, "temp_tokens", i)
        temp_files.append(output_file)
        print(f"Tokenized and saved chunk {i+1}/{num_chunks} to {output_file}")

        del contents_slice
        gc.collect()

    bm25 = qutil.build_bm25_from_disk(temp_files)
    return bm25


## Union Method
def hybrid_search(query, bm25, bm25_doc_ids, faiss_index, retrieval_model, faiss_doc_ids, top_k=10):
    """
    Run a hybrid BM25 + FAISS search for one query.
    Returns the list of retrieved doc IDs (in rank order).
    """
    # 1) FAISS dense retrieval
    q_emb = retrieval_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = faiss_index.search(q_emb, top_k)
    faiss_hits = [faiss_doc_ids[i] for i in I[0]]
    
    # 2) BM25 sparse retrieval
    tokenized_q = query.lower().split()
    scores = bm25.get_scores(tokenized_q)
    top_idxs = np.argsort(scores)[::-1][:top_k]
    bm25_hits = [bm25_doc_ids[i] for i in top_idxs]
    
    # 3) Merge, preserving FAISS order first
    seen, hybrid = set(), []
    for did in faiss_hits + bm25_hits:
        if did not in seen:
            seen.add(did)
            hybrid.append(did)
    return hybrid


## Evaluate Union Hybrid
def evaluate_hybrid_end_to_end(
    bm25, 
    bm25_doc_ids,
    faiss_index,
    doc_contents,
    doc_ids,
    retrieval_model,
    questions_path,
    top_k: int = 10
) -> float:
    with open(questions_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    qa_pairs = []
    for i in range(0, len(lines), 3):
        qa_pairs.append({
            "question": lines[i+1],
            "answer": lines[i+2].lower()
        })
    
    # Start Evaluation
    correct = 0
    mrr = 0
    for idx, qa in enumerate(qa_pairs, 1):
        hits = hybrid_search(
            qa["question"], 
            bm25, 
            bm25_doc_ids,
            faiss_index, 
            retrieval_model,
            doc_ids,
            top_k=top_k
        )
        # check if answer string appears in any retrieved doc
        text_hits = (doc_contents[did].lower() for did in hits)
        found = False
        for rank, doc in enumerate(text_hits, start=1):
            if qa["answer"] in doc:
                correct += 1
                mrr += 1 / rank
                print("✅", correct, idx + 1, f"(Rank: {rank}), MRR:{mrr}")
                found = True
                break

        if not found:
            print("❌", correct, idx + 1, f"(Rank:--), MRR:{mrr}")

    
    recall = correct / len(qa_pairs)
    print(f"\n  Recall@{top_k}: {recall:.3%}")
    return recall


## Sequential Method
def sequential_hybrid_search(
    query: str,
    bm25,
    bm25_doc_ids,
    retrieval_model,
    full_docs,
    top_k=10
):
    """
    Step 1: BM25 retrieves candidates.
    Step 2: FAISS reranks those BM25 candidates.
    (Optionally add CrossEncoder later.)
    """
    # Step 1: BM25 retrieval
    tokenized_q = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_q)
    top_idxs = np.argsort(bm25_scores)[::-1][:top_k*10]
    bm25_hits = [bm25_doc_ids[i] for i in top_idxs]

    # Step 2: FAISS semantic reranking (only over BM25 hits)
    candidate_texts = [full_docs[doc_id] for doc_id in bm25_hits]
    candidate_embs = retrieval_model.encode(candidate_texts, convert_to_numpy=True)
    query_emb = retrieval_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(candidate_embs)
    faiss.normalize_L2(query_emb)

    # Compute cosine similarity manually
    similarities = np.dot(candidate_embs, query_emb.T).squeeze()
    reranked = [bm25_hits[i] for i in np.argsort(similarities)[::-1]]

    return reranked[:top_k]


## Evaluate Sequential 
def evaluate_sequential_hybrid(
    bm25,
    bm25_doc_ids: list,
    retrieval_model,
    full_docs: dict,
    questions_path: str,
    top_k: int = 10
):
    """
    Load questions from questions.txt and evaluate Sequential Hybrid Retrieval.
    Print detailed results and final summary.
    """
    # --- Load Questions ---
    with open(questions_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    questions = []
    for i in range(0, len(lines), 3):
        category = lines[i]
        question = lines[i+1]
        answer = lines[i+2]
        questions.append((category, question, answer))

    # --- Evaluate ---
    total, rr_list, correct_at_1, correct_at_k = len(questions), [], 0, 0
    mrr = 0

    for idx, (category, question, answer) in enumerate(questions, start=1):
        query = f"{category} : {question}"
        hits = sequential_hybrid_search(
            query, bm25, bm25_doc_ids,
            retrieval_model = retrieval_model, 
            full_docs=full_docs,
            top_k=top_k, 
        )

        # Match answer
        low_answer = answer.lower()
        found_rank = None
        for rank, doc_id in enumerate(hits, start=1):
            if low_answer in full_docs[doc_id].lower():
                found_rank = rank
                break

        # Collect stats
        if found_rank == 1:
            correct_at_1 += 1
        if found_rank and found_rank <= top_k:
            correct_at_k += 1
            rr_list.append(1.0 / found_rank)
            mrr += 1/found_rank
            status = f"✅ Rank {found_rank}"
        else:
            rr_list.append(0.0)
            status = "❌"

        # Print per-question
        print(f"    → Expected: {answer}")
        print(f"{status} [{category}] Q: {question}")
        print(f"MRR : {mrr}")
        print()

    # --- Final Metrics ---
    mrr = sum(rr_list) / total
    p_at_1 = correct_at_1 / total
    recall_k = correct_at_k / total

    print("―" * 60)
    print(f"📊 MRR         : {mrr:.4f}")
    print(f"📊 Precision@1 : {p_at_1:.4f}")
    print(f"📊 Recall@{top_k}  : {recall_k:.4f} ({correct_at_k}/{total})")
    print(f"📊 Total Qs    : {total}")

    return {"MRR": mrr, "P@1": p_at_1, f"Recall@{top_k}": recall_k}




