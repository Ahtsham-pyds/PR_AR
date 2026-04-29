import faiss
import numpy as np
import pickle
import os

from sentence_transformers import SentenceTransformer

# -----------------------------
# MODEL
# -----------------------------
model = SentenceTransformer("C:/Users/hahtsham/work/PR_PO/PR_AR/models/all-MiniLM-L6-v2")
# -----------------------------
# CONFIG
# -----------------------------
DIMENSION = 384
INDEX_FILE = "faiss.index"
META_FILE = "faiss_meta.pkl"

# -----------------------------
# LOAD / INIT INDEX
# -----------------------------
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)

else:
    index = faiss.IndexFlatL2(DIMENSION)
    metadata = []
    
    
def add_to_vector_store(text: str, extra_meta=None):

    embedding = model.encode([text])
    embedding = np.array(embedding).astype("float32")

    index.add(embedding)

    metadata.append({
        "text": text,
        "meta": extra_meta
    })

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
        
        
        
def vector_search(query: str, top_k=5):

    if len(metadata) == 0:
        return []

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx]["text"])

    return results