import faiss
import numpy as np
import os

EMBEDDINGS_PATH = "embeddings"

def build_faiss_index():
    embeddings = np.load(os.path.join(EMBEDDINGS_PATH, "embeddings.npy"))

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(EMBEDDINGS_PATH, "faiss_index.bin"))

    print("✅ FAISS index created!")