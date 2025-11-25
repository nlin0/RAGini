"""
vector_db.py builds and queries a vector database with FAISS.
This is Component 2 of the RAG pipeline - Vector Search.
"""
import os
# fix OpenMP conflict on macOS when using both FAISS and PyTorch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import faiss
import numpy as np

PREPROCESSED_FILE = "preprocessed_documents.json"
INDEX_FILE = "faiss.index"


def load_preprocessed_documents(filename):
    """Load preprocessed documents from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def build_faiss_index(embeddings):
    """
    Build a FAISS index from document embeddings.
    
    Args:
        embeddings: List of embedding vectors or numpy array of shape (n_docs, 768)
        
    Returns:
        FAISS index ready for search
    """
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


class VectorDB:
    """Vector database for similarity search using FAISS."""
    
    def __init__(self, preprocessed_file=PREPROCESSED_FILE):
        """
        Initialize the vector database.
        
        Args:
            preprocessed_file: Path to preprocessed_documents.json
        """
        print("Loading preprocessed documents...")
        self.documents = load_preprocessed_documents(preprocessed_file)
        
        print("Building FAISS index...")
        embeddings = [doc["embedding"] for doc in self.documents]
        self.index = build_faiss_index(embeddings)
        
        print(f"Vector database ready with {len(self.documents)} documents.")
    
    def search(self, query_embedding, top_k=3):
        """
        Search for top-k most similar documents.
        
        Args:
            query_embedding: numpy array of shape (768,) - query vector
            top_k: Number of results to return
            
        Returns:
            tuple: (distances, indices) where:
                - distances: numpy array of shape (top_k,) - L2 distances
                - indices: numpy array of shape (top_k,) - document indices
        """
        # reshape to (1, 768) for FAISS batch search
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        
        # search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # return as 1D arrays
        return distances[0], indices[0]
    
    def get_document_by_index(self, idx):
        """Get document by its index in the database."""
        return self.documents[idx]


if __name__ == "__main__":
    # test the vector database
    from encode import QueryEncoder
    
    print("Initializing vector database...")
    db = VectorDB()
    
    print("\nInitializing query encoder...")
    encoder = QueryEncoder()
    
    # test search
    test_query = "What causes squirrels to lose fur?"
    print(f"\nQuery: {test_query}")
    
    query_emb = encoder.encode(test_query)
    distances, indices = db.search(query_emb, top_k=5)
    
    print("\nTop Results:")
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        doc = db.get_document_by_index(idx)
        print(f"{i+1}. (ID {doc['id']}) Distance={dist:.4f}")
        print(f"   {doc['text'][:100]}...")
        print()
