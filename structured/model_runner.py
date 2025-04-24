# === Unified Model Runner Framework (OOP) ===
# Purpose: Modular and extensible design with shared base logic for custom models

import argparse
import os
import pickle
import importlib
from abc import ABC, abstractmethod

# === Shared utilities ===
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# === Abstract base class ===
class RetrievalModel(ABC):
    def __init__(self, name):
        self.name = name
        self.doc_contents = load_pickle('doc_contents.pkl') if os.path.exists('doc_contents.pkl') else {}
        self.doc_titles = load_pickle('doc_titles.pkl') if os.path.exists('doc_titles.pkl') else {}

    @abstractmethod
    def index(self):
        pass

    @abstractmethod
    def query(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


# === Example: Dense FAISS model ===
class DenseRetrievalModel(RetrievalModel):
    def index(self):
        import embed
        import combine_chunks
        import faiss_vector_build

    def query(self):
        import query_test

    def evaluate(self):
        import batch_query_test


# === Example: Sparse TF-IDF model ===
class SparseRetrievalModel(RetrievalModel):
    def index(self):
        import indexing
        import calculate

    def query(self):
        import search

    def evaluate(self):
        import search  # reuse search + custom evaluation if needed


# === Example: Custom Model (e.g., BERT + TF-IDF) ===
class HybridRetrievalModel(RetrievalModel):
    def index(self):
        import indexing
        import embed
        import combine_chunks
        import faiss_vector_build
        import calculate

    def query(self):
        import hybrid_query  # should be implemented by user

    def evaluate(self):
        import hybrid_batch_eval  # should be implemented by user


# === Registry for available models ===
MODEL_REGISTRY = {
    'dense': DenseRetrievalModel,
    'sparse': SparseRetrievalModel,
    'hybrid': HybridRetrievalModel,
    # Add new ones here
}


def main():
    parser = argparse.ArgumentParser(description="Run retrieval models")
    parser.add_argument('--model', required=True, help='Model type: dense | sparse | hybrid')
    parser.add_argument('--test', action='store_true', help='Run evaluation instead of full indexing/query pipeline')
    args = parser.parse_args()

    model_type = args.model.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' not recognized.")

    model = MODEL_REGISTRY[model_type](model_type)

    if args.test:
        model.evaluate()
    else:
        model.index()
        model.query()


if __name__ == '__main__':
    main()
