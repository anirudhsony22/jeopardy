import pickle
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
import pickle_utils as putil
import query_utils as qutil
import time

start_time = time.time()

base_path = '../database'
questions_path = '../questions.txt'
opt = "bge_"
UNION = False
SEQ_E2B = False
SEQ_E2E = True
FAISS_RECALL_TEST = False

# Load documents
doc_contents = putil.open_pickle(os.path.join(base_path, "doc_contents.pkl"))
doc_titles = putil.open_pickle(os.path.join(base_path, f"{opt}doc_titles.pkl"))
doc_ids = putil.open_pickle(os.path.join(base_path, f"{opt}doc_ids.pkl"))
# doc_ids = np.load(os.path.join(base_path, "doc_ids.npy"))
# doc_ids = sorted(doc_contents.keys())[:subset_size]

## Need embeddings and faiss only for hybrid union query
doc_embeddings = np.load(os.path.join(base_path, f"{opt}doc_embeddings.npy"))
faiss_index = faiss.read_index(os.path.join(base_path, f"{opt}faiss_index.idx"))

# Load embedding model for question embedding
# embedding_model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
embedding_model_name = "BAAI/bge-base-en-v1.5"
retrieval_model = SentenceTransformer(embedding_model_name)


# Parameters for BM25 embedding
subset_size = 50000  # Use only 50k docs
chunk_size = 5000    # Tokenize 5000 docs at a time

# Subsample contents
subset_contents = {doc_id: doc_contents[doc_id] for doc_id in doc_ids}
titles = putil.open_pickle(os.path.join(base_path, "doc_titles.pkl"))
bm25, bm25_doc_ids = putil.open_pickle(os.path.join(base_path, "bm25_cache.pkl"))

if UNION:
    ##Union method
    recall10 = qutil.evaluate_hybrid_end_to_end(
    bm25=bm25,
    bm25_doc_ids=bm25_doc_ids,
    faiss_index=faiss_index,
    doc_contents=doc_contents,
    doc_ids=doc_ids,
    retrieval_model=retrieval_model,
    questions_path=questions_path,
    top_k=10
    )

if SEQ_E2B:
    ##Sequential Method
    # seq_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    seq_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    metrics = qutil.evaluate_sequential_hybrid(
        retrieval_model=seq_model,
        bm25=bm25,
        bm25_doc_ids=bm25_doc_ids,
        faiss_index=faiss_index,
        full_docs=doc_contents,
        faiss_doc_ids=doc_ids,
        questions_path=questions_path,
        top_k=10,
        faiss_top_k=10000,
        bm25_top_k=10,
        faiss_first=True
    )

if SEQ_E2E:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu')
    filter_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    qutil.evaluate_sequential_encoding(
        retrieval_model=filter_model,
        cross_encoder=cross_encoder,
        faiss_index=faiss_index,
        docs_content=doc_contents,
        faiss_doc_ids=doc_ids,
        questions_path=questions_path,
        top_k=10,
        faiss_top_k=100,
        )


if FAISS_RECALL_TEST:
    qutil.evaluate_faiss_only(faiss_index=faiss_index,
                            faiss_doc_ids=doc_ids,
                            doc_contents=doc_contents,
                            retrieval_model=retrieval_model,
                            questions_path=questions_path,
                            top_k=100
                            )


end_time = time.time()
print(f"⏱️ Execution time: {end_time - start_time:.4f} seconds")