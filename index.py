import numpy as np

class SimpleVectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.embeddings = []
        self.texts = []

    def add(self, text, embedding):
        self.texts.append(text)
        self.embeddings.append(embedding)

    def search(self, query_embedding, top_k=3):
        if not self.embeddings:
            return []

        embeddings = np.array(self.embeddings)
        query = np.array(query_embedding)

        # Cosine similarity
        dot_products = np.dot(embeddings, query)
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query)
        similarities = dot_products / norms

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.texts[i], float(similarities[i])) for i in top_indices]
