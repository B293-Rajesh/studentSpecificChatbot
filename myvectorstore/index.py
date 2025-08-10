import numpy as np
import pickle
import os

class SimpleVectorStore:
    def _init_(self, dim, index_path="vector_index.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.vectors = []
        self.metadata = []

        if os.path.exists(index_path):
            self._load()

    def add(self, vectors, metas):
        for v, m in zip(vectors, metas):
            vec = np.array(v, dtype=np.float32)
            if vec.shape[0] != self.dim:
                raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {vec.shape[0]}")
            self.vectors.append(vec)
            self.metadata.append(m)
        self._save()

    def search(self, query_vector, k=3, metric="l2"):
        query_vector = np.array(query_vector, dtype=np.float32)
        if metric == "l2":
            dists = [np.linalg.norm(vec - query_vector) for vec in self.vectors]
        elif metric == "cosine":
            dists = [1 - np.dot(vec, query_vector) / (np.linalg.norm(vec) * np.linalg.norm(query_vector) + 1e-10) for vec in self.vectors]
        else:
            raise ValueError("Unsupported metric. Use 'l2' or 'cosine'.")

        sorted_idx = np.argsort(dists)
        top_k_idx = sorted_idx[:k]
        return [(self.metadata[i], dists[i]) for i in top_k_idx]

    def _save(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump((self.vectors, self.metadata), f)

    def _load(self):
        with open(self.index_path, 'rb') as f:
            self.vectors, self.metadata = pickle.load(f)
