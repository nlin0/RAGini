"""
prompt_augmentation.py combines user queries with retrieved documents into structured prompts.
This is Component 4 of the RAG pipeline - Prompt Augmentation.
"""


def augment_prompt(query, retrieved_documents, top_k=3):
    """
    Create an augmented prompt combining the query with retrieved context.
    
    Args:
        query: User's natural language question
        retrieved_documents: List of document text strings (top-k retrieved documents)
        top_k: Number of documents to include (default 3)
        
    Returns:
        str: Augmented prompt ready for LLM generation
    """
    # limit to top_k documents
    documents = retrieved_documents[:top_k]
    
    # build the augmented prompt
    prompt = f"Question: {query}\n\n"
    prompt += "Top documents:\n"
    
    for i, doc_text in enumerate(documents, 1):
        prompt += f"{i}. {doc_text}\n"
    
    prompt += "\nBased on the context above, provide a detailed answer."
    
    return prompt


def format_prompt_for_chat(query, retrieved_documents, top_k=3):
    """
    Format prompt for chat-based models (like TinyLlama, Qwen, Llama3.2).
    Uses a more conversational format.
    
    Args:
        query: User's natural language question
        retrieved_documents: List of document text strings
        top_k: Number of documents to include
        
    Returns:
        str: Formatted prompt for chat models
    """
    documents = retrieved_documents[:top_k]
    
    # build context section
    context = "\n\n".join([f"[Document {i+1}]: {doc}" for i, doc in enumerate(documents)])
    
    # format as a chat-style prompt
    prompt = f"""Use the following documents to answer the question. If the answer cannot be found in the documents, say so.

Documents:
{context}

Question: {query}

Answer:"""
    
    return prompt


if __name__ == "__main__":
    # test prompt augmentation
    test_query = "What causes squirrels to lose fur?"
    test_docs = [
        "Squirrels can lose fur due to various reasons including parasites, stress, or seasonal changes.",
        "Fur loss in squirrels is often caused by mites or other external parasites that affect the skin.",
        "Nutritional deficiencies can also lead to fur loss in squirrels, especially during winter months."
    ]
    
    print("Test Query:", test_query)
    print("\n" + "="*60)
    print("Augmented Prompt:")
    print("="*60)
    print(augment_prompt(test_query, test_docs))
    
    print("\n" + "="*60)
    print("Chat-Formatted Prompt:")
    print("="*60)
    print(format_prompt_for_chat(test_query, test_docs))


