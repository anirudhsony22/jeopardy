import pickle
import math
from collections import defaultdict

def nested_defaultdict():
    return defaultdict(int)

# === Load inverted index and doc count ===
with open('./inverted_index.pkl', 'rb') as f:
    inverted_index = pickle.load(f)

N = len(set(doc_id for term in inverted_index for doc_id in inverted_index[term]))

# === Step 1: Compute IDF ===
idf = {}
for term, doc_freqs in inverted_index.items():
    df = len(doc_freqs)
    idf[term] = math.log(N / df)

# === Step 2: Compute TF-IDF and Document Norms ===
doc_norms = defaultdict(float)

for term, doc_freqs in inverted_index.items():
    for doc_id, tf in doc_freqs.items():
        tfidf = tf * idf[term]
        doc_norms[doc_id] += tfidf ** 2

# Finalize norms by taking sqrt
for doc_id in doc_norms:
    doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])

# === Save results ===
with open('./idf.pkl', 'wb') as f:
    pickle.dump(idf, f)

with open('./doc_norms.pkl', 'wb') as f:
    pickle.dump(dict(doc_norms), f)
