import pickle
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
import pickle_utils as putil
import query_utils as qutil

file_path = './'
questions_path = './questions.txt'

# Load documents
base_path = file_path
doc_contents = putil.open_pickle(os.path.join(base_path, "doc_contents.pkl"))
doc_titles = putil.open_pickle(os.path.join(base_path, "doc_titles.pkl"))
doc_ids = np.load(os.path.join(base_path, "doc_ids.pkl"))
# doc_ids = sorted(doc_contents.keys())[:subset_size]
doc_embeddings = np.load(os.path.join(base_path, "doc_embeddings.npy"))
faiss_index = faiss.read_index(os.path.join(base_path, "faiss_index.idx"))

# Load embedding model for question embedding
embedding_model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
retrieval_model = SentenceTransformer(embedding_model_name)


# Parameters for BM25 embedding
subset_size = 50000  # Use only 50k docs
chunk_size = 5000    # Tokenize 5000 docs at a time

# Subsample contents

subset_contents = {doc_id: doc_contents[doc_id] for doc_id in doc_ids}
titles = putil.open_pickle(os.path.join(base_path, "doc_titles.pkl"))
bm25, bm25_doc_ids = putil.open_pickle(os.path.join(base_path, "bm25_cache_50k.pkl"))


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

##Sequential Method
seq_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
metrics = qutil.evaluate_sequential_hybrid(
    bm25=bm25,
    bm25_doc_ids=bm25_doc_ids,
    retrieval_model=seq_model,
    full_docs=doc_contents,
    doc_titles=titles,
    questions_path='/content/jeopardy_project/questions.txt',
    top_k=10,
)


