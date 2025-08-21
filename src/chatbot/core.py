# src/chatbot/core.py

import os
# Ensure the .llm_interface import is correct for relative paths
from .llm_interface import OllamaClient
# Import the Retriever from the rag sub-package
from .rag.retriever import Retriever 

class Chatbot:
    def __init__(self, model_name: str = "Qwen/Qwen1.5-1.8B-Chat"):
        """
        Initializes the main chatbot logic.
        Args:
            model_name (str): The name of the LLM model to use.
        """
        print(f"Initializing Chatbot with LLM model: {model_name}")
        self.llm_client = OllamaClient(model_name=model_name)
        self.chat_history = [] 

        # --- NEW RAG SYSTEM INITIALIZATION ---
        self.rag_system = Retriever(
            db_path="data/chroma_db",
            collection_name="nilecare_knowledge" 
        )
        
        print("Attempting to load knowledge base from 'data/knowledge_base.txt'...")
        self.rag_system.vector_db.load_knowledge_base("data/knowledge_base.txt")
        print("Knowledge base loaded into RAG system.")
        # --- END NEW RAG SYSTEM INITIALIZATION ---

        print("Chatbot initialized with RAG system and knowledge base loaded.")

    def process_message(self, user_message: str) -> str:
        """
        Processes a user's message with a hybrid RAG/general-knowledge approach.
        1. It tries to find a relevant document in the knowledge base.
        2. If relevant documents are found, it uses them to augment the prompt.
        3. If no relevant documents are found, it falls back to the LLM's general knowledge
           and adds a disclaimer about the information's source.
        """
        print(f"You: {user_message}")

        # --- RAG Step 1: Retrieve relevant information ---
        # We retrieve with a lower similarity threshold to be flexible
        retrieved_docs = self.rag_system.retrieve_info(user_message, n_results=3, min_similarity=0.4)
        
        # This will hold the final prompt we send to the LLM
        full_prompt = ""
        
        # Check if any documents were actually retrieved
        if retrieved_docs:
            print("\n--- RAG Context Provided to LLM ---")
            context_str = "\n\nRelevant Information from Knowledge Base:\n"
            for i, doc in enumerate(retrieved_docs):
                context_str += f"--- Document {i+1} ---\n"
                context_str += doc['content'] + "\n"
            context_str += "\n"
            print(context_str)
            print("-----------------------------------")
            
            # This is the RAG-augmented prompt
            full_prompt = (
                f"You are a helpful and informative healthcare chatbot. "
                f"Using *only* the following information from the knowledge base, answer the user's question concisely and accurately. "
                f"If the question cannot be answered from the provided information, or if it requires personalized medical advice or diagnosis, "
                f"state clearly that you don't have enough information to answer that question and strongly suggest consulting a qualified healthcare professional.\n\n"
                f"{context_str}"
                f"User's Question: {user_message}"
            )
        else:
            print("\n--- No relevant context found. Falling back to general knowledge. It might not be vey accurate ---")
            # This is the general-knowledge prompt with the transparency clause
            full_prompt = (
                f"You are a helpful and informative chatbot. Please answer the user's question to the best of your ability. "
                f"If the question is about a specific health condition or requires a diagnosis, you must add a disclaimer "
                f"stating that the information is from your general knowledge and that the user should consult a qualified "
                f"healthcare professional for an accurate diagnosis.\n\n"
                f"User's Question: {user_message}"
            )

        # Send the constructed prompt to the LLM
        response = self.llm_client.generate_response(full_prompt, self.chat_history)

        # Update chat history
        self.chat_history.append({'role': 'user', 'content': user_message})
        self.chat_history.append({'role': 'assistant', 'content': response})

        print(f"Chatbot: {response}")
        return response

    def reset_conversation(self):
        """Resets the chat history, effectively starting a new conversation."""
        self.chat_history = []
        print("Chat history reset. Starting a fresh conversation!")

if __name__ == "__main__":
    print(f"\n--- Testing Chatbot Core (Hybrid RAG Enabled) ---")
    test_chatbot = Chatbot()
    print("\n--- Simulating Conversation ---")
    # This query should use the RAG system
    test_chatbot.process_message("Tell me about having a balanced diet.")
    # This query should fall back to general knowledge and add a disclaimer
    test_chatbot.process_message("What are some common symptoms of strep throat?")
