# import nltk
# nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
import json
from tqdm import tqdm

input_path = "../database/cleaned_paragraph_chunks.jsonl"
output_path = "../database/nltk_chunking.jsonl"

with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in tqdm(fin, desc="Chunking into sentences"):
        data = json.loads(line)
        doc_id = data["doc_id"]
        title = data["title"]
        paragraph = data["chunk_text"]
        para_idx = data["position"]

        # Sentence tokenization
        sentences = sent_tokenize(paragraph)

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent.split()) >= 5:  # skip short junk
                fout.write(json.dumps({
                    "doc_id": doc_id,
                    "title": title,
                    "proposition_text": sent,
                    "parent_paragraph": para_idx,
                    "prop_id": f"{doc_id}_p{para_idx}_s{i}"
                }) + "\n")