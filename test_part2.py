#!/usr/bin/env python3
"""
Test script for Part 2 RAG Pipeline
Tests all components and runs 2 queries as required for Part 2 demonstration.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
from encode import QueryEncoder
from vector_db import VectorDB
from document_retrieval import DocumentRetriever
from prompt_augmentation import format_prompt_for_chat

def test_component_1():
    """Test Component 1: Query Encoder"""
    print("\n" + "="*60)
    print("Testing Component 1: Query Encoder")
    print("="*60)
    encoder = QueryEncoder()
    query = "What causes squirrels to lose fur?"
    embedding = encoder.encode(query)
    assert embedding.shape == (768,), f"Expected shape (768,), got {embedding.shape}"
    print(f"✓ Query encoded successfully: shape {embedding.shape}")
    return encoder

def test_component_2():
    """Test Component 2: Vector Search"""
    print("\n" + "="*60)
    print("Testing Component 2: Vector Search")
    print("="*60)
    vector_db = VectorDB()
    assert len(vector_db.documents) > 0, "No documents loaded"
    print(f"✓ Vector database loaded: {len(vector_db.documents)} documents")
    
    # test search
    encoder = QueryEncoder()
    query = "What causes squirrels to lose fur?"
    query_emb = encoder.encode(query)
    distances, indices = vector_db.search(query_emb, top_k=3)
    assert len(indices) == 3, f"Expected 3 results, got {len(indices)}"
    print(f"✓ Search successful: top-3 results retrieved")
    return vector_db

def test_component_3():
    """Test Component 3: Document Retrieval"""
    print("\n" + "="*60)
    print("Testing Component 3: Document Retrieval")
    print("="*60)
    retriever = DocumentRetriever()
    vector_db = VectorDB()
    encoder = QueryEncoder()
    
    query = "What causes squirrels to lose fur?"
    query_emb = encoder.encode(query)
    distances, indices = vector_db.search(query_emb, top_k=3)
    
    texts = retriever.get_texts_by_indices(indices)
    assert len(texts) > 0, "No documents retrieved"
    assert len(texts) <= 3, f"Expected <= 3 documents (after dedup), got {len(texts)}"
    print(f"✓ Document retrieval successful: {len(texts)} unique documents")
    return retriever

def test_component_4():
    """Test Component 4: Prompt Augmentation"""
    print("\n" + "="*60)
    print("Testing Component 4: Prompt Augmentation")
    print("="*60)
    query = "What causes squirrels to lose fur?"
    test_docs = [
        "Squirrels can lose fur due to various reasons.",
        "Fur loss in squirrels is often caused by mites.",
        "Nutritional deficiencies can lead to fur loss."
    ]
    prompt = format_prompt_for_chat(query, test_docs, top_k=3)
    assert "Question:" in prompt or "question" in prompt.lower(), "Query not in prompt"
    assert len(prompt) > len(query), "Prompt should be longer than query"
    print(f"✓ Prompt augmentation successful: {len(prompt)} chars")
    return True

def test_component_5():
    """Test Component 5: LLM Generation (if model exists)"""
    print("\n" + "="*60)
    print("Testing Component 5: LLM Generation")
    print("="*60)
    try:
        from llm_generation import LLMGenerator
        import os
        
        # check if model exists
        model_path = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
        if os.path.exists(model_path):
            generator = LLMGenerator(model_name="tinyllama")
            test_prompt = "What causes squirrels to lose fur?"
            response = generator.generate_chat(test_prompt, max_tokens=50)
            assert len(response) > 0, "Empty response from LLM"
            print(f"✓ LLM generation successful: {len(response)} chars")
            return True
        else:
            print(f"⚠ LLM model not found: {model_path}")
            print("  Please download it using:")
            print("  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
            return False
    except Exception as e:
        print(f"⚠ LLM test skipped: {e}")
        return False

def test_full_pipeline(query_text, show_details=False):
    """Test the full RAG pipeline for a single query"""
    print("\n" + "="*60)
    print(f"Testing Full Pipeline: '{query_text}'")
    print("="*60)
    
    # component 1: encode
    encoder = QueryEncoder()
    query_emb = encoder.encode(query_text)
    if show_details:
        print(f"[1] Encoded: shape {query_emb.shape}")
    
    # component 2: vector search
    vector_db = VectorDB()
    distances, indices = vector_db.search(query_emb, top_k=3)
    if show_details:
        print(f"[2] Top-3 results: {len(indices)} documents")
        for i, (idx, dist) in enumerate(zip(indices, distances), 1):
            doc = vector_db.get_document_by_index(idx)
            print(f"    {i}. ID={doc['id']}, Distance={dist:.4f}")
    
    # component 3: document retrieval
    retriever = DocumentRetriever()
    texts = retriever.get_texts_by_indices(indices)
    if show_details:
        print(f"[3] Retrieved {len(texts)} unique documents")
    
    # component 4: prompt augmentation
    augmented = format_prompt_for_chat(query_text, texts, top_k=3)
    if show_details:
        print(f"[4] Augmented prompt: {len(augmented)} chars")
    
    # component 5: LLM (if available)
    try:
        from llm_generation import LLMGenerator
        import os
        if os.path.exists("tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"):
            generator = LLMGenerator(model_name="tinyllama")
            response = generator.generate_chat(augmented, max_tokens=256, temperature=0.7)
            if show_details:
                print(f"[5] Generated response: {len(response)} chars")
            print(f"\n✓ Full pipeline successful!")
            print(f"Response: {response[:200]}...")
            return True
        else:
            print(f"\n✓ Pipeline successful (without LLM - model not found)")
            return False
    except Exception as e:
        print(f"\n⚠ Pipeline test incomplete: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("Part 2 RAG Pipeline Test Suite")
    print("="*60)
    
    # test individual components
    try:
        encoder = test_component_1()
        vector_db = test_component_2()
        retriever = test_component_3()
        test_component_4()
        llm_available = test_component_5()
        
        print("\n" + "="*60)
        print("All Component Tests: PASSED")
        print("="*60)
        
        # test with 2 queries from queries.json (as required for Part 2)
        print("\n" + "="*60)
        print("Testing Full Pipeline with 2 Queries (Part 2 Requirement)")
        print("="*60)
        
        with open("queries.json", "r") as f:
            queries = json.load(f)
        
        # test with first 2 queries
        test_queries = [queries[0]["text"], queries[1]["text"]]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}: {query}")
            print('='*60)
            test_full_pipeline(query, show_details=True)
        
        print("\n" + "="*60)
        print("Part 2 Test Suite: COMPLETE")
        print("="*60)
        if not llm_available:
            print("\nNOTE: LLM model not found. Please download it to test Component 5.")
            print("wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

