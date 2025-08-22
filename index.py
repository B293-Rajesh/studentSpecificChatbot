import numpy as np

class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.text_ids = []

    def add(self, vector, text_id):
        self.vectors.append(vector)
        self.text_ids.append(text_id)

    def search(self, query_vector, top_k=3):
        if not self.vectors:
            return []

        vectors = np.array(self.vectors)
        query_vector = np.array(query_vector)

        # Cosine similarity
        sims = np.dot(vectors, query_vector) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        )
        top_k_idx = np.argsort(sims)[-top_k:][::-1]
        return [self.text_ids[i] for i in top_k_idx]
