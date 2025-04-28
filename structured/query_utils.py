import os
import gc
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
import torch


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
def union_hybrid_search(query, bm25, bm25_doc_ids, faiss_index, retrieval_model, faiss_doc_ids, top_k=10):
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
        hits = union_hybrid_search(
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


## Sequential Method (BM25->FAISS)
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

## Smart sequential (FAISS->BM25)
def smart_hybrid_search_restricted_optimized(
    query,
    faiss_index,
    faiss_doc_ids,
    retrieval_model,
    doc_contents,
    faiss_top_k=10000,
    bm25_top_k=10
):
    """
    Retrieve top-k documents using FAISS first,
    then rerank the candidates using a temporary BM25 built on-the-fly.
    """

    # --- Step 1: FAISS retrieval ---
    q_emb = retrieval_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    print(q_emb.shape[1], faiss_index.d)

    D, I = faiss_index.search(q_emb, faiss_top_k)
    faiss_hits = [faiss_doc_ids[i] for i in I[0]]
    
    # --- Step 2: Tokenize FAISS hits ---
    candidate_texts = [doc_contents[doc_id] for doc_id in faiss_hits]
    tokenized_candidates = [text.lower().split() for text in candidate_texts]
    
    # --- Step 3: Build temporary BM25 ---
    bm25_subset = BM25Okapi(tokenized_candidates)
    
    # --- Step 4: BM25 retrieval among candidates ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25_subset.get_scores(tokenized_query)
    top_idxs = np.argsort(bm25_scores)[::-1][:bm25_top_k]
    reranked_hits = [faiss_hits[i] for i in top_idxs]

    return reranked_hits

## Evaluation for Sequential and Smart Sequential
def evaluate_sequential_hybrid(
    retrieval_model,
    bm25,
    bm25_doc_ids: list,
    faiss_index,
    full_docs: dict,
    faiss_doc_ids,
    questions_path: str,
    top_k: int = 10,
    faiss_top_k=10000,
    bm25_top_k=10,
    faiss_first = True,
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
        if faiss_first:
            hits = smart_hybrid_search_restricted_optimized(
                    query,
                    faiss_index,
                    faiss_doc_ids,
                    retrieval_model,
                    full_docs,
                    faiss_top_k=faiss_top_k,
                    bm25_top_k=bm25_top_k
                )
        else:
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



## FAISS->Cross-Encoding
def evaluate_sequential_encoding(
    retrieval_model,
    cross_encoder,
    faiss_index,
    docs_content: dict,
    faiss_doc_ids,
    questions_path: str,
    top_k: int = 10,
    faiss_top_k=1000,
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
        q_emb = retrieval_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = faiss_index.search(q_emb, faiss_top_k)
        faiss_hits = [faiss_doc_ids[i] for i in I[0]]

        hits = rerank_single_query_topk(query, faiss_hits, cross_encoder, docs_content, batch_size=32, top_k=top_k)

        # Match answer
        low_answer = answer.lower()
        found_rank = None
        for rank, doc_id in enumerate(hits, start=1):
            if low_answer in docs_content[doc_id].lower():
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


## Sequential Encoders
def rerank_single_query_topk(
    query, 
    candidate_doc_list, 
    cross_encoder,
    docs_content, 
    batch_size=64,
    top_k=20,
):
    cross_inputs = [(query, docs_content[doc_id]) for doc_id in candidate_doc_list]

    scores = []
    cross_encoder.model.eval()
    with torch.no_grad():
        for start_idx in range(0, len(cross_inputs), batch_size):
            batch = cross_inputs[start_idx:start_idx + batch_size]
            batch_scores = cross_encoder.predict(batch)
            scores.extend(batch_scores)
    
    # Rerank documents based on scores
    scores = np.array(scores)
    sorted_idx = np.argsort(-scores)  # descending order
    
    # Select top-k
    topk_idx = sorted_idx[:top_k]
    topk_docs = [candidate_doc_list[i] for i in topk_idx]
    
    return topk_docs

## For FAISS Recall calculation (Experimenting)
def evaluate_faiss_only(
    faiss_index,
    faiss_doc_ids,
    doc_contents,
    retrieval_model,
    questions_path,
    top_k=500
):
    # Load QA pairs
    with open(questions_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    qa_pairs = []
    for i in range(0, len(lines), 3):
        qa_pairs.append({
            "question": lines[i+1],
            "answer": lines[i+2].lower()
        })
    
    correct = 0
    mrr = 0
    for idx, qa in enumerate(qa_pairs, 1):
        # Encode query
        q_emb = retrieval_model.encode([qa["question"]], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = faiss_index.search(q_emb, top_k)
        faiss_hits = [faiss_doc_ids[i] for i in I[0]]
        
        # Check if answer appears
        text_hits = (doc_contents[did].lower() for did in faiss_hits)
        found = False
        for rank, doc in enumerate(text_hits, start=1):
            if qa["answer"] in doc:
                correct += 1
                mrr += 1 / rank
                print(f"✅ {correct}/{idx} (Rank: {rank})")
                found = True
                break
        
        if not found:
            print(f"❌ {correct}/{idx} (Rank: --)")
    
    recall = correct / len(qa_pairs)
    print(f"\n📈 FAISS Recall@{top_k}: {recall:.2%}")
    print(f"📈 FAISS MRR@{top_k}: {mrr / len(qa_pairs):.4f}")
    return recall
