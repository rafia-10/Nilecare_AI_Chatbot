import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLMInterface:
    def __init__(self, model_name: str):
        """
        Initializes the LLM interface using a Hugging Face transformers model.
        Args:
            model_name (str): The name of the Hugging Face model to load.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        print(f"Loading Hugging Face model: {self.model_name}")
        try:
            # Load the tokenizer and model from Hugging Face Hub.
            # 'torch_dtype="auto"' selects the best dtype available (e.g., float16 on GPU).
            # 'device_map="auto"' distributes the model across available devices (GPU/CPU).
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512
            )
            print("Hugging Face model loaded successfully.")
        except Exception as e:
            # Provide an informative error message for the user.
            print(f"Failed to load Hugging Face model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load Hugging Face model {self.model_name}: {e}")

    def generate_response(self, prompt: str, chat_history: list = None) -> str:
        """
        Generates a response from the LLM based on a prompt and chat history.
        Args:
            prompt (str): The user's current prompt, which can include RAG context.
            chat_history (list): A list of dictionaries representing previous messages.
                                  Each dict: {'role': 'user'/'assistant', 'content': 'message'}.
        Returns:
            str: The generated response from the LLM.
        """
        if chat_history is None:
            chat_history = []

        # Construct the full message list for the LLM, including history and the new prompt.
        messages = chat_history + [{'role': 'user', 'content': prompt}]

        try:
            # The transformers pipeline handles the conversation turn.
            response = self.pipeline(
                messages,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
            
            # The response is a list of dictionaries. We extract the generated content.
            generated_text = response[0]['generated_text']
            
            # Find the last message content, which is the model's response.
            # This is robust because the pipeline returns the full conversation history.
            if isinstance(generated_text, list) and generated_text:
                return generated_text[-1].get('content', '')
            else:
                return str(generated_text)
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I am unable to generate a response at this time."

# --- Test the LLMInterface ---
if __name__ == "__main__":
    
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    
    print(f"Attempting to connect to Hugging Face with model: {model_name}")

    try:
        llm_client = LLMInterface(model_name=model_name)
    except RuntimeError as e:
        print(f"Exiting due to initialization error: {e}")
        exit(1)

    if llm_client:
        print("\n--- Testing Response Generation ---")
        initial_prompt = "Hello, tell me a bit about general health advice."
        print(f"User: {initial_prompt}")
        response = llm_client.generate_response(initial_prompt)
        print(f"Chatbot: {response}")

        print("\n--- Testing with a follow-up (simulated chat history) ---")
        chat_history = [
            {'role': 'user', 'content': initial_prompt},
            {'role': 'assistant', 'content': response}
        ]
        follow_up_prompt = "What are some common symptoms of the flu?"
        print(f"User: {follow_up_prompt}")
        response_follow_up = llm_client.generate_response(follow_up_prompt, chat_history)
        print(f"Chatbot: {response_follow_up}")

    print("\nTest complete.")
