import pickle
from collections import defaultdict

def nested_defaultdict():
    return defaultdict(int)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def open_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret