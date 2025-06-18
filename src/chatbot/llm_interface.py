import ollama
import os

class OllamaClient:
    def __init__(self, model_name: str = "llama3:latest"):
        """
        Initializes the Ollama client with a specific model.
        Args:
            model_name (str): The name of the model to use (e.g., 'llama3:latest').
        """
        self.model_name = model_name
        try:
            # Test if the Ollama server is reachable and the model exists
            # This call might raise an exception if the server is down or model is not found
            ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': 'Hi'}], stream=False, options={'num_predict': 1})
            print(f"Ollama client initialized successfully with model: {self.model_name}")
        except ollama.ResponseError as e:
            print(f"Error initializing Ollama client: {e}")
            print(f"Please ensure Ollama server is running and model '{self.model_name}' is downloaded.")
            exit(1) # Exit if we can't connect to Ollama

    def generate_response(self, prompt: str, chat_history: list = None) -> str:
        """
        Generates a response from the LLM based on the prompt and chat history.
        Args:
            prompt (str): The user's current input.
            chat_history (list): A list of dictionaries representing previous messages.
                                  Each dict: {'role': 'user'/'assistant', 'content': 'message'}.
        Returns:
            str: The generated response from the LLM.
        """
        if chat_history is None:
            chat_history = []

        # Add the current user prompt to the chat history
        messages = chat_history + [{'role': 'user', 'content': prompt}]

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=False, 
                options={'num_predict': 256, 'temperature': 0.6}
            )
            
            return response['message']['content']
        except ollama.ResponseError as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

# --- Test the OllamaClient ---
if __name__ == "__main__":
    
    from dotenv import load_dotenv
    load_dotenv() # Load environment variables from .env file

    model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")

    print(f"Attempting to connect to Ollama with model: {model_name}")

    ollama_client = OllamaClient(model_name=model_name)

    if ollama_client:
        print("\n--- Testing Response Generation ---")
        initial_prompt = "Hello, tell me a bit about general health advice."
        print(f"User: {initial_prompt}")
        response = ollama_client.generate_response(initial_prompt)
        print(f"Chatbot: {response}")

        print("\n--- Testing with a follow-up (simulated chat history) ---")
        # Simulate a basic chat history
        chat_history = [
            {'role': 'user', 'content': initial_prompt},
            {'role': 'assistant', 'content': response}
        ]
        follow_up_prompt = "What are some common symptoms of the flu?"
        print(f"User: {follow_up_prompt}")
        response_follow_up = ollama_client.generate_response(follow_up_prompt, chat_history)
        print(f"Chatbot: {response_follow_up}")

    print("\nTest complete.")