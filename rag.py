import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

EMB_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "data/faiss.index"
DOCS_FILE = "data/docs.json"

model = SentenceTransformer(EMB_MODEL)


def index_docs(docs):
    """docs: list of {'id': str, 'text': str}"""
    texts = [d['text'] for d in docs]
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    json.dump(docs, open(DOCS_FILE, 'w'))
    return True


def load_index():
    if not os.path.exists(INDEX_FILE):
        return None, None
    index = faiss.read_index(INDEX_FILE)
    docs = json.load(open(DOCS_FILE, 'r'))
    return index, docs


def rag_query(query, top_k=3):
    index, docs = load_index()
    if index is None:
        return "No documents indexed."
    emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = index.search(emb, top_k)
    results = []
    for idx in I[0]:
        if idx < len(docs):
            results.append(docs[idx])
    # return concatenated context — in real app you'd call an LLM to produce final answer
    return " | ".join([r['text'] for r in results])
