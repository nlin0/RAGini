"""
document_retrieval.py handles mapping document IDs back to document texts.
This is Component 3 of the RAG pipeline - Document Retrieval.
"""
import json

PREPROCESSED_FILE = "preprocessed_documents.json"


class DocumentRetriever:
    """In-memory document retriever using dictionary lookup."""
    
    def __init__(self, preprocessed_file=PREPROCESSED_FILE):
        """
        Initialize the document retriever.
        
        Args:
            preprocessed_file: Path to preprocessed_documents.json
        """
        with open(preprocessed_file, "r") as f:
            documents = json.load(f)
        
        # build in-memory dictionary: index -> document
        self.documents_by_index = documents
        
        # mapping by document id (in case multiple chunks have same id)
        self.documents_by_id = {}
        for idx, doc in enumerate(documents):
            doc_id = doc["id"]
            if doc_id not in self.documents_by_id:
                self.documents_by_id[doc_id] = []
            self.documents_by_id[doc_id].append((idx, doc))
    
    def get_by_index(self, index):
        """
        Get document by its index in the preprocessed array.
        
        Args:
            index: Integer index in the preprocessed documents array
            
        Returns:
            dict with keys: 'id', 'text', 'embedding'
        """
        if 0 <= index < len(self.documents_by_index):
            return self.documents_by_index[index]
        return None
    
    def get_by_id(self, doc_id):
        """
        Get all document chunks with a given document ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of (index, document) tuples
        """
        return self.documents_by_id.get(doc_id, [])
    
    def get_texts_by_indices(self, indices):
        """
        Get document texts for a list of indices.
        
        Args:
            indices: List of integer indices
            
        Returns:
            List of document text strings
        """
        texts = []
        seen_ids = set()  # deduplication by document ID
        
        for idx in indices:
            doc = self.get_by_index(idx)
            if doc is None:
                continue
            
            # deduplicate: if we've seen this document ID before, skip it
            if doc["id"] in seen_ids:
                continue
            
            seen_ids.add(doc["id"])
            texts.append(doc["text"])
        
        return texts


if __name__ == "__main__":
    # test
    # retriever = DocumentRetriever()
    # 
    # # test by index
    # print("\nTesting retrieval by index:")
    # doc = retriever.get_by_index(0)
    # if doc:
    #     print(f"Document ID: {doc['id']}")
    #     print(f"Text (first 100 chars): {doc['text'][:100]}...")
    # 
    # # test by multiple indices
    # print("\nTesting retrieval by multiple indices:")
    # texts = retriever.get_texts_by_indices([0, 1, 2, 3, 4])
    # print(f"Retrieved {len(texts)} unique documents")
    # for i, text in enumerate(texts):
    #     print(f"{i+1}. {text[:80]}...")
    pass


