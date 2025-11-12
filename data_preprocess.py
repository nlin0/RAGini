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

# constants that can be changed later on if needed
MODEL_NAME = "BAAI/bge-base-en-v1.5"
INPUT_FILE = "documents.json"
OUTPUT_FILE = "preprocessed_documents.json"


def embed_docs(docs, model, tokenizer, device='cpu'):
    """
    Embed documents using PyTorch and the BGE model.
    
    Parameters:
        docs: List of documents (dicts or strings)
        model: PyTorch model for encoding
        tokenizer: Tokenizer for the model
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        List of processed documents with embeddings
    """
    processed = []
    model.eval()  # set model to evaluation mode

    with torch.no_grad():  # disable gradient computation for inference
        for i, doc in enumerate(tqdm(docs, desc='Embedding documents')):
            # determine text for document
            if type(doc) is dict:
                if "text" in doc:
                    text = doc["text"]
                else:
                    text = ""
            else:
                # document is just a plain string
                text = doc

            # tokenize and encode the text
            encoded_input = tokenizer(text, max_length=512, padding=True, 
                                     truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            model_output = model(**encoded_input)
            # BGE model, mean pooling over token embeddings and take the [CLS] token embedding
            embeddings = model_output.last_hidden_state
            # mean pooling: average over sequence length dimension
            attention_mask = encoded_input['attention_mask']
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
            
            # l2 normalization
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embedding = embeddings.cpu().squeeze().tolist()

            # get document ID
            if type(doc) is dict and "id" in doc:
                doc_id = doc["id"]
            else:
                doc_id = i

            # add to processed list
            processed.append({
                'id': doc_id,
                "text": text,
                "embedding": embedding
            })

    return processed


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    
    with open(INPUT_FILE, 'r') as f:
        docs = json.load(f)
    
    processed = embed_docs(docs, model, tokenizer, device)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(processed, f)
    
    print(f"Processed {len(processed)} documents and saved to {OUTPUT_FILE}")