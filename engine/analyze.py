import pickle
from collections import Counter, defaultdict

def nested_defaultdict():
    return defaultdict(int)

with open('./inverted_index.pkl', 'rb') as f:
    inverted_index = pickle.load(f)
with open('./doc_lengths.pkl', 'rb') as f:
    doc_lengths = pickle.load(f)
with open('./doc_titles.pkl', 'rb') as f:
    doc_titles = pickle.load(f)


print(f"Number of Documents: {len(doc_titles)}")
print(f"Vocabulary Size: {len(inverted_index)}")

# === Count total term frequency across corpus ===
term_frequencies = Counter()

for term, doc_freqs in inverted_index.items():
    total_term_freq = sum(doc_freqs.values())
    term_frequencies[term] = total_term_freq


