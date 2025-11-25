"""
encode.py converts user questions into 768-dimensional vectors using the BGE model.
This is Component 1 of the RAG pipeline.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np

MODEL_NAME = "BAAI/bge-base-en-v1.5"

def get_device():
    """
    Decide the best available device.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class QueryEncoder:
    """
    Encodes natural language queries into 768-dimensional vectors.
    """
    
    def __init__(self, model_name=MODEL_NAME, device=None):
        """
        Initialize the encoder with BGE model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (auto-detected if None)
        """
        if device is None:
            device = get_device()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def encode(self, query_text):
        """
        Encode a query text into a 768-dimensional vector.
        
        Args:
            query_text: Natural language query string
            
        Returns:
            numpy array of shape (768,) - the query embedding
        """
        # tokenize
        encoded = self.tokenizer(
            query_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # forward pass
        with torch.no_grad():
            out = self.model(**encoded)
            seq = out.last_hidden_state  # [1, seq_len, 768]
            mask = encoded["attention_mask"]
            
            # mean pooling
            pooled = (seq * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
            
            # normalize
            emb = F.normalize(pooled, p=2, dim=1)
        
        # return numpy array shape
        return emb.cpu().numpy().squeeze().astype("float32")


def encode_query(query_text, encoder=None):
    """
    Convenience function to encode a query.
    
    Args:
        query_text: Natural language query string
        encoder: QueryEncoder instance (creates new one if None)
        
    Returns:
        numpy array of shape (768,) - the query embedding
    """
    if encoder is None:
        encoder = QueryEncoder()
    return encoder.encode(query_text)


if __name__ == "__main__":
    # encoder test
    encoder = QueryEncoder()
    
    test_query = "What causes squirrels to lose fur?"
    embedding = encoder.encode(test_query)
    
    print(f"Query: {test_query}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 values): {embedding[:10]}")


