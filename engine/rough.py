import pickle

with open('doc_contents.pkl', 'rb') as f:
    doc_contents = pickle.load(f)

print("Total documents:", len(doc_contents))
