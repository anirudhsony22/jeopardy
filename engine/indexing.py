import re
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
import os

def nested_defaultdict():
    return defaultdict(int)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
folder_path = '../wikidata/'
files = os.listdir(folder_path)
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

def preprocess(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [ps.stem(t) for t in tokens if t not in stop_words]

def build_index(files):
    inverted_index = defaultdict(nested_defaultdict)
    doc_lengths = {}
    doc_titles = {}
    doc_contents = {}
    doc_id = 0
    for file in files:
        print(file, doc_id)
        with open(file, 'r', encoding='utf-8') as f:
            current_title = None
            content_lines = []
            for line in f:
                line = line.strip()
                if line.startswith('[[') and line.endswith(']]'):
                    # Save the previous document before moving to the next
                    if current_title is not None:
                        full_text = ' '.join(content_lines)
                        tokens = preprocess(current_title + " " + full_text)
                        doc_lengths[doc_id] = len(tokens)
                        doc_titles[doc_id] = current_title
                        doc_contents[doc_id] = full_text
                        for term in tokens:
                            inverted_index[term][doc_id] += 1
                        doc_id += 1
                    
                    # Start new document
                    current_title = line[2:-2].strip()
                    # print(current_title)
                    content_lines = []
                else:
                    content_lines.append(line)
            
            # Save last document
            if current_title is not None:
                full_text = ' '.join(content_lines)
                tokens = preprocess(current_title + " " + full_text)
                doc_lengths[doc_id] = len(tokens)
                doc_titles[doc_id] = current_title
                doc_contents[doc_id] = full_text
                for term in tokens:
                    inverted_index[term][doc_id] += 1
    
    with open('./inverted_index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)

    with open('./doc_titles.pkl', 'wb') as f:
        pickle.dump(doc_titles, f)

    with open('./doc_lengths.pkl', 'wb') as f:
        pickle.dump(dict(doc_lengths), f)
 
    with open('./doc_contents.pkl', 'wb') as f:
        pickle.dump(doc_contents, f)


    return inverted_index, doc_lengths, doc_titles, doc_contents

build_index([folder_path+file for file in files])