import os
from dotenv import load_dotenv
from src.chatbot.core import Chatbot

def main():
    load_dotenv()
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen:1.8b-chat")
    print(f"Starting NileCare Chatbot with model: {model_name}")

    # Initialize the chatbot
    chatbot = Chatbot(model_name=model_name)

    print("\n--- NileCare Chatbot (Type 'exit' to quit, 'reset' to clear conversation) ---")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        elif user_input.lower() == 'reset':
            chatbot.reset_conversation()
            continue

        # Process the user's message
        chatbot.process_message(user_input)

if __name__ == "__main__":
    main()