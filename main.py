import os
import sys
from src.chatbot.core import Chatbot

def main():
    """
    Main function to run the NileCare Chatbot.
    Initializes the chatbot and starts the conversational loop.
    """
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    print(f"Starting NileCare Chatbot with model: {model_name}")

    try:
        # Initialize our Chatbot instance
        chatbot = Chatbot(model_name=model_name)

        print("\n--- Welcome to NileCare Chatbot ---")
        print("Type 'exit' to quit the conversation.")
        print("Type 'reset' to clear the current conversation history and start fresh.")
        print("Type your message below and press Enter.")
        print("-----------------------------------")

        # Main conversational loop
        while True:
            try:
                user_input = input("You: ") # Get input from the user

                if user_input.lower() == 'exit':
                    print("Exiting chatbot. Goodbye! Thank you for using NileCare Chatbot.")
                    break # Exit the loop and end the program
                elif user_input.lower() == 'reset':
                    chatbot.reset_conversation() # Call the chatbot's reset method
                    continue # Skip to the next loop iteration (ask for new input)
                
                # Process the user's message using the chatbot's core logic
                # The response will be printed by chatbot.process_message
                chatbot.process_message(user_input)

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\nExiting chatbot. Goodbye!")
                break
            except Exception as e:
                # Catch any unexpected errors during interaction
                print(f"An unexpected error occurred: {e}")
                print("Please try again or type 'exit' to quit.")
                # It might be wise to break here if the error is critical to prevent an infinite loop
                break

    except Exception as e:
        # Catch errors during chatbot initialization (e.g., model loading failure)
        print(f"\nAn error occurred during chatbot setup: {e}")
        print("Please ensure all dependencies are installed and the model is accessible.")

if __name__ == "__main__":
    main()
