"""
main.py - Interactive RAG System
This is Component 6 of the RAG pipeline - The Main Program.

Orchestrates the complete RAG workflow:
1. Query Encoding
2. Vector Search
3. Document Retrieval
4. Prompt Augmentation
5. LLM Generation
"""
import os
# fix OpenMP conflict on macOS when using both FAISS and PyTorch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from encode import QueryEncoder
from vector_db import VectorDB
from document_retrieval import DocumentRetriever
from prompt_augmentation import format_prompt_for_chat
from llm_generation import LLMGenerator

# configuration
TOP_K = 3  # number of documents to retrieve
PREPROCESSED_FILE = "preprocessed_documents.json"


class RAGSystem:
    """Complete RAG pipeline system."""
    
    def __init__(self, model_name="tinyllama", model_path=None):
        """
        Initialize all RAG components.
        
        Args:
            model_name: Name of LLM model to use
            model_path: Direct path to model file (overrides model_name)
        """
        print("="*60)
        print("Initializing RAG System")
        print("="*60)
        
        # component 1: query encoder
        print("\n[1/5] Loading Query Encoder...")
        self.encoder = QueryEncoder()
        
        # component 2: vector database
        print("\n[2/5] Loading Vector Database...")
        self.vector_db = VectorDB(PREPROCESSED_FILE)
        
        # component 3: document retriever
        print("\n[3/5] Loading Document Retriever...")
        self.doc_retriever = DocumentRetriever(PREPROCESSED_FILE)
        
        # component 4: prompt augmentation (no initialization needed)
        print("\n[4/5] Prompt Augmentation ready")
        
        # component 5: LLM generator
        print("\n[5/5] Loading LLM Generator...")
        try:
            if model_path:
                self.llm = LLMGenerator(model_path=model_path)
            else:
                self.llm = LLMGenerator(model_name=model_name)
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print("\nPlease download a model first. For example:")
            print("wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("RAG System Ready!")
        print("="*60)
    
    def process_query(self, query, top_k=TOP_K, show_context=False):
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: Natural language question
            top_k: Number of documents to retrieve
            show_context: Whether to display retrieved documents
            
        Returns:
            str: Generated answer
        """
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # step 1: encode query
        print("\n[Step 1] Encoding query...")
        query_embedding = self.encoder.encode(query)
        print(f"Query encoded to vector of shape {query_embedding.shape}")
        
        # step 2: vector search
        print(f"\n[Step 2] Searching for top-{top_k} documents...")
        distances, indices = self.vector_db.search(query_embedding, top_k=top_k)
        print(f"Found {len(indices)} results")
        
        if show_context:
            print("\nRetrieved documents:")
            for i, (idx, dist) in enumerate(zip(indices, distances), 1):
                doc = self.vector_db.get_document_by_index(idx)
                print(f"\n  [{i}] (ID: {doc['id']}, Distance: {dist:.4f})")
                print(f"  {doc['text'][:150]}...")
        
        # step 3: document retrieval
        print(f"\n[Step 3] Retrieving document texts...")
        retrieved_texts = self.doc_retriever.get_texts_by_indices(indices)
        print(f"Retrieved {len(retrieved_texts)} unique documents")
        
        # step 4: prompt augmentation
        print(f"\n[Step 4] Augmenting prompt with context...")
        augmented_prompt = format_prompt_for_chat(query, retrieved_texts, top_k=top_k)
        
        if show_context:
            print("\nAugmented prompt (first 500 chars):")
            print(augmented_prompt[:500] + "...")
        
        # step 5: LLM generation
        print(f"\n[Step 5] Generating response with LLM...")
        response = self.llm.generate_chat(augmented_prompt, max_tokens=256, temperature=0.7)
        
        return response
    
    def interactive_loop(self):
        """Run interactive command-line interface."""
        print("\n" + "="*60)
        print("Interactive RAG System")
        print("="*60)
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type 'quit' or 'exit' to exit")
        print("  - Type 'show' to toggle showing retrieved context")
        print("="*60 + "\n")
        
        show_context = False
        
        while True:
            try:
                query = input("Query: ").strip()
                
                if query.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break
                
                if query.lower() == "show":
                    show_context = not show_context
                    print(f"Show context: {'ON' if show_context else 'OFF'}")
                    continue
                
                if not query:
                    continue
                
                # process the query
                response = self.process_query(query, top_k=TOP_K, show_context=show_context)
                
                # display response
                print(f"\n{'='*60}")
                print("Answer:")
                print('='*60)
                print(response)
                print('='*60)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive RAG System")
    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        choices=["tinyllama", "llama3.2", "qwen2-1.5b", "qwen2-7b"],
        help="LLM model to use"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Direct path to GGUF model file (overrides --model)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to process (non-interactive mode)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of documents to retrieve (default: {TOP_K})"
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved documents in output"
    )
    
    args = parser.parse_args()
    
    # initialize system
    rag = RAGSystem(model_name=args.model, model_path=args.model_path)
    
    # run in interactive or single-query mode
    if args.query:
        # single query mode
        response = rag.process_query(args.query, top_k=args.top_k, show_context=args.show_context)
        print(f"\n{'='*60}")
        print("Answer:")
        print('='*60)
        print(response)
        print('='*60)
    else:
        # interactive mode
        rag.interactive_loop()


if __name__ == "__main__":
    main()


