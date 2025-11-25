"""
llm_generation.py handles LLM inference for text generation.
This is Component 5 of the RAG pipeline - LLM Generation.
"""
from llama_cpp import Llama
import os

# model options (user should download one of these)
MODEL_OPTIONS = {
    "tinyllama": "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
    "llama3.2": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "qwen2-1.5b": "qwen2-1_5b-instruct-q4_0.gguf",
    "qwen2-7b": "qwen2-7b-instruct-q4_0.gguf"
}


class LLMGenerator:
    """Wrapper for LLM text generation using llama-cpp-python."""
    
    def __init__(self, model_path=None, model_name="tinyllama", n_ctx=2048, n_threads=None):
        """
        Initialize the LLM generator.
        
        Args:
            model_path: Direct path to GGUF model file (overrides model_name)
            model_name: Name of model to use (tinyllama, llama3.2, qwen2-1.5b, qwen2-7b)
            n_ctx: Context window size
            n_threads: Number of threads (None = auto-detect)
        """
        if model_path is None:
            if model_name not in MODEL_OPTIONS:
                raise ValueError(f"Unknown model_name: {model_name}. Choose from {list(MODEL_OPTIONS.keys())}")
            model_path = MODEL_OPTIONS[model_name]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please download the model first. For example:\n"
                f"wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
            )
        
        print(f"Loading LLM model: {model_path}")
        print("This may take a moment...")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        
        self.model_name = model_name
        print("LLM model loaded successfully!")
    
    def generate(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9, stop=None):
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            stop: List of stop sequences (None for default)
            
        Returns:
            str: Generated text
        """
        # for chat models, we might want to use a chat format
        # but for simplicity, we'll use direct generation
        # different models may need different prompt formats
        
        if stop is None:
            # default stop sequences
            stop = ["\n\nQuestion:", "\n\n---", "Question:"]
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False  # don't echo the prompt
        )
        
        return output["choices"][0]["text"].strip()
    
    def generate_chat(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
        """
        Generate text with chat-style formatting.
        Some models work better with specific chat formats.
        
        Args:
            prompt: Input prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            str: Generated text
        """
        # for TinyLlama and similar chat models, we can use a simple format
        # more sophisticated models might need system/user/assistant format
        
        full_prompt = f"### Human: {prompt}\n### Assistant:"
        
        output = self.llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["### Human:", "\n###", "###", "\n\nQuestion:", "\nQuestion:", "Question:"],
            echo=False
        )
        
        return output["choices"][0]["text"].strip()


if __name__ == "__main__":
    # test LLM generation
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "tinyllama"
    
    try:
        print(f"Testing LLM generation with model: {model_name}")
        generator = LLMGenerator(model_name=model_name)
        
        test_prompt = "What causes squirrels to lose fur?"
        print(f"\nPrompt: {test_prompt}")
        print("\nGenerating response...")
        
        response = generator.generate(test_prompt, max_tokens=128)
        print(f"\nResponse:\n{response}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download a model first. For TinyLlama:")
        print("wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")


