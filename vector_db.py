"""
vector_db.py builds and queries a simple vector database with FAISS. It loads the
precomputed document embeddings, builds a FAISS index for similarity search,
and allows users to query index with a natural language.
"""
import json
import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
PREPROCESSED_FILE = "preprocessed_documents.json"
INDEX_FILE = "faiss.index"

def load_preprocessed_documents(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_faiss_index(embeddings, dim):
    index = faiss.IndexFlatL2(dim)
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    return index

def save_faiss_index(index, index_path):
    faiss.write_index(index, index_path)

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def embed_query(text, model, tokenizer, device='cpu'):
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        encoded_input = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        output = model(**encoded_input)
        embeddings = output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().squeeze().numpy().astype('float32')

def query_index(query, index, model, tokenizer, documents, top_k=5, device='cpu'):
    query_embedding = embed_query(query, model, tokenizer, device=device)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    D, I = index.search(query_embedding, top_k)
    # D: distances, I: indices
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(documents):
            continue
        doc = documents[idx]
        doc_copy = doc.copy() if isinstance(doc, dict) else {"text": str(doc)}
        doc_copy["score"] = float(dist)
        results.append(doc_copy)
    return results

def main():

    # Load preprocessed documents and embeddings
    documents = load_preprocessed_documents(PREPROCESSED_FILE)
    embeddings = [doc["embedding"] for doc in documents]
    dim = len(embeddings[0])

    # Build FAISS index if not saved, otherwise load existing
    try:
        index = load_faiss_index(INDEX_FILE)
    except Exception:
        index = build_faiss_index(embeddings, dim)
        save_faiss_index(index, INDEX_FILE)

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    while True:
        query = input("Query: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break
        results = query_index(query, index, model, tokenizer, documents, top_k=5, device=device)
        print("\nTop Results:")
        for i, res in enumerate(results):
            print(f"{i+1}. {res.get('text', '')} (Score: {res['score']:.4f})")
        print("\n")

if __name__ == "__main__":
    main()
