"""
vector_db.py builds and queries a simple vector database with FAISS. It loads the
precomputed document embeddings, builds a FAISS index for similarity search,
and allows users to query index with a natural language.
"""
import json
import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

MODEL_NAME = "BAAI/bge-base-en-v1.5"
PREPROCESSED_FILE = "preprocessed_documents.json"
INDEX_FILE = "faiss.index"

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_preprocessed_documents(filename):
    with open(filename, "r") as f:
        return json.load(f)


def build_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def embed_query(query, model, tokenizer, device):
    enc = tokenizer(
        query, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        seq = out.last_hidden_state
        mask = enc["attention_mask"]

        pooled = (seq * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
        pooled = F.normalize(pooled, p=2, dim=1)

    return pooled.cpu().numpy().astype("float32")  # shape (1, 768)


def query_index(query, index, model, tokenizer, documents, top_k=5, device="cpu"):
    q_emb = embed_query(query, model, tokenizer, device)
    D, I = index.search(q_emb, top_k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        doc = documents[idx]
        results.append({
            "id": doc["id"],
            "text": doc["text"],
            "score": float(dist)
        })
    return results


def main():
    device = get_device()
    print(f"Using device: {device}")

    # print("Loading documents...")
    documents = load_preprocessed_documents(PREPROCESSED_FILE)
    embeddings = [doc["embedding"] for doc in documents]

    # print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Loading BGE model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    while True:
        query = input("Query: ").strip()
        if query in ["quit", "exit"]:
            break

        results = query_index(query, index, model, tokenizer, documents, top_k=5, device=device)
        
        print("\nTop Results:")
        for i, r in enumerate(results):
            print(f"{i+1}. (ID {r['id']}) Score={r['score']:.4f}")
            print(r["text"])
            print("")

if __name__ == "__main__":
    main()
