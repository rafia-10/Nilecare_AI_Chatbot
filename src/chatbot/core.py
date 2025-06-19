from .llm_interface import OllamaClient # Import our Ollama client
import os

class Chatbot:
    def __init__(self, model_name: str = "qwen:1.8b-chat"): 
        """
        Initializes the main chatbot logic.
        """
        print(f"Initializing Chatbot with LLM model: {model_name}")
        self.llm_client = OllamaClient(model_name=model_name)
        self.chat_history = [] 

    def process_message(self, user_message: str) -> str:
        """
        Processes a user's message, interacts with the LLM, and returns a response.
        This is where the 'brain' logic will grow (NLU, RAG, etc.).
        """
        #print(f"User: {user_message}")

        # For now, we'll just send the user message directly to the LLM
        # In future steps, we'll add NLU, RAG, and conditional logic here.
        response = self.llm_client.generate_response(user_message, self.chat_history)

        # Update chat history (important for maintaining context)
        self.chat_history.append({'role': 'user', 'content': user_message})
        self.chat_history.append({'role': 'assistant', 'content': response})

        print(f"Chatbot: {response}")
        return response

    def reset_conversation(self):
        """Resets the chat history for a new conversation."""
        self.chat_history = []
        print("Chat history reset.")