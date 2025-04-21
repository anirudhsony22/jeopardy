from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# === Load test questions ===
input_file = '../questions.txt'  # change this to your file path
with open(input_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

assert len(lines) % 3 == 0, "File format error: expected triplets of CATEGORY, QUESTION, ANSWER"

# === Load system components ===
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('faiss_index.idx')

with open('doc_ids.pkl', 'rb') as f:
    doc_ids = pickle.load(f)
with open('doc_titles.pkl', 'rb') as f:
    doc_titles = pickle.load(f)
with open('doc_contents.pkl', 'rb') as f:
    doc_contents = pickle.load(f)

# === Run test ===
correct = 0
total = 0

for i in range(0, len(lines), 3):
    category = lines[i]
    question = lines[i + 1]
    answer = lines[i + 2]

    # Embed and search
    query_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_embedding, k=1)
    top_idx = I[0][0]
    doc_id = doc_ids[top_idx]
    retrieved_text = doc_contents.get(doc_id, "").lower()

    # Check if answer is present in result (case-insensitive substring match)
    found = answer.lower() in retrieved_text
    total += 1
    if found:
        correct += 1
        result = "✅"
    else:
        result = "❌"

    print(f"{result} [{category}] Q: {question}")
    print(f"    ➤ Expected: {answer}")
    print(f"    ➤ Matched Doc Title: {doc_titles.get(doc_id, '???')}")
    print()

# === Summary ===
accuracy = correct / total * 100
print(f"\n📊 Accuracy: {correct}/{total} = {accuracy:.2f}%")
