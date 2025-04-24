import re, os
from collections import defaultdict
import pickle_utils as pkl
import argparse

def parse_wikidata_txt(input_folder_path: str, output_folder_path: str):
    doc_contents, doc_titles = {}, {}
    doc_id = 0

    for file in sorted(os.listdir(input_folder_path)):
        if not file.endswith('.txt'): continue
        with open(os.path.join(input_folder_path, file), 'r', encoding='utf-8') as f:
            current_title, content_lines = None, []
            for line in f:
                line = line.strip()
                if line.startswith('[[') and line.endswith(']]'):
                    if current_title is not None:
                        doc_contents[doc_id] = ' '.join(content_lines)
                        doc_titles[doc_id] = current_title
                        doc_id += 1
                    current_title = line[2:-2].strip()
                    content_lines = []
                else:
                    content_lines.append(line)
            if current_title is not None:
                doc_contents[doc_id] = ' '.join(content_lines)
                doc_titles[doc_id] = current_title
                doc_id += 1
        print(f"Docs Parsed: {doc_id}")

    pkl.save_pickle(doc_contents, f'{output_folder_path}/doc_contents.pkl')
    pkl.save_pickle(doc_titles, f'{output_folder_path}/doc_titles.pkl')

    print(f"Saved {len(doc_contents)} documents in {output_folder_path}")   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse .txt files into document chunks")
    parser.add_argument('--input', default='../wikidata', help='Path to folder with raw .txt files')
    parser.add_argument('--output', default='../database', help='Output folder for .pkl files')
    args = parser.parse_args()

    parse_wikidata_txt(args.input, args.output)