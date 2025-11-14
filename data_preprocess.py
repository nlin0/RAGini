"""
data_preprocess.py is used to preprocess the data for the model. Specifically,
it reads the dataset, encodes the document through the BGE model,
and outputs the encoded document and the query to a JSON file..
"""
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
INPUT_FILE = "documents.json"
OUTPUT_FILE = "preprocessed_documents.json"

def get_device():
    """
    Decide the best available device:
      - Apple M1/M2 → 'mps'
      - CUDA GPU → 'cuda'
      - Otherwise → 'cpu'
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_docs(docs, model, tokenizer, device):
    processed = []
    model.eval()

    with torch.no_grad():
        for doc in tqdm(docs, desc="Encoding documents"):
            text = doc.get("text", "")
            doc_id = doc.get("id")

            # tokenization
            encoded = tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # forward pass
            out = model(**encoded)

            # mean pool
            seq = out.last_hidden_state  # [1, seq_len, 768]
            mask = encoded["attention_mask"]
            pooled = (seq * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)

            # normalize
            emb = F.normalize(pooled, p=2, dim=1)
            embedding = emb.cpu().squeeze().tolist()

            processed.append({
                "id": doc_id,
                "text": text,
                "embedding": embedding
            })

    return processed


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        docs = json.load(f)

    processed = embed_docs(docs, model, tokenizer, device)

    print(f"Saving {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(processed, f)

    print(f"Done! Encoded {len(processed)} documents.")
