import pickle
import math
import re
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def nested_defaultdict():
    return defaultdict(int)

# === Setup Preprocessing ===
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [ps.stem(t) for t in tokens if t not in stop_words]

# === Load all data ===
with open('./inverted_index.pkl', 'rb') as f:
    inverted_index = pickle.load(f)
with open('./idf.pkl', 'rb') as f:
    idf = pickle.load(f)
with open('./doc_norms.pkl', 'rb') as f:
    doc_norms = pickle.load(f)
with open('./doc_titles.pkl', 'rb') as f:
    doc_titles = pickle.load(f)

# === Main Query Function ===
def search(query, top_k=5):
    query_terms = preprocess(query)
    query_tf = Counter(query_terms)
    
    # Step 1: Compute query vector (TF-IDF)
    query_vec = {}
    for term, tf in query_tf.items():
        if term in idf:
            query_vec[term] = tf * idf[term]
    query_norm = math.sqrt(sum(w**2 for w in query_vec.values()))
    if query_norm == 0:
        return []

    # Step 2: Score documents (only those that share terms with query)
    scores = defaultdict(float)
    for term, q_wt in query_vec.items():
        if term not in inverted_index:
            continue
        for doc_id, doc_tf in inverted_index[term].items():
            doc_wt = doc_tf * idf[term]
            scores[doc_id] += q_wt * doc_wt

    # Step 3: Normalize scores
    for doc_id in scores:
        scores[doc_id] /= (query_norm * doc_norms[doc_id])

    # Step 4: Rank and return top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_titles[doc_id], score) for doc_id, score in ranked[:top_k]]

# === Example Usage ===
if __name__ == "__main__":
    
    query = input()
    while query!='exit':
        results = search(query, top_k=5)
        print("\nTop Results:")
        for title, score in results:
            print(f"{title} (score: {score:.4f})")

        print()
        print()
        query = input()
