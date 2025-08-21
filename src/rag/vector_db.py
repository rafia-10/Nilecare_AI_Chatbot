# src/chatbot/rag/vector_db.py

import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import uuid

class VectorDB:
    def __init__(self, knowledge_file_path: str):
        """
        Initializes the VectorDB with a ChromaDB client and a sentence transformer for embeddings.
        Args:
            knowledge_file_path (str): The path to the knowledge base text file.
        """
        self.knowledge_file_path = knowledge_file_path
        
        # Determine the ChromaDB directory from the knowledge file path
        # This makes the path relative to the project structure
        db_directory = os.path.dirname(os.path.dirname(self.knowledge_file_path))
        if not db_directory:
            db_directory = os.path.join(os.getcwd(), 'data', 'chroma_db')
        else:
            db_directory = os.path.join(db_directory, 'chroma_db')

        # Create the ChromaDB client to connect to the database.
        # This will store the database files in the specified directory.
        self.client = chromadb.PersistentClient(path=db_directory)

        # Initialize the SentenceTransformer model.
        # This model converts text into numerical vectors (embeddings).
        # We use a specific model that's good for semantic search.
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create or get a collection in the database. A collection is where documents are stored.
        # We use a simple name for our knowledge base.
        self.collection_name = "nilecare_knowledge"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            # We must specify the embedding function to match our SentenceTransformer model.
            embedding_function=self._get_embedding_function()
        )

        print(f"VectorDB initialized. ChromaDB path: {db_directory}")

    def _get_embedding_function(self):
        """
        Returns a custom embedding function to wrap the SentenceTransformer model.
        ChromaDB needs a function that takes a list of texts and returns a list of embeddings.
        """
        def embed_documents(texts: List[str]) -> List[List[float]]:
            """Converts a list of texts to a list of embeddings."""
            # The .encode() method from SentenceTransformer does exactly what we need.
            return self.embedding_model.encode(texts).tolist()

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def load_knowledge_base(self):
        """
        Reads the knowledge base file, chunks the text, and stores it in the vector database.
        """
        print(f"Loading knowledge base from: {self.knowledge_file_path}")
        try:
            with open(self.knowledge_file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except FileNotFoundError:
            print(f"Error: Knowledge base file not found at {self.knowledge_file_path}")
            return

        # Split the text into documents based on the "---" delimiter
        documents = raw_text.strip().split("---")
        
        # Clear the collection to avoid duplicate data on re-runs
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._get_embedding_function()
        )
        print("Existing knowledge base cleared from database.")
        
        texts = []
        metadatas = []
        ids = []

        for doc_text in documents:
            if doc_text.strip():
                # Extract the first line as the title
                lines = doc_text.strip().split("\n")
                title = lines[0].strip() if lines else "Untitled"
                content = "\n".join(lines[1:]).strip()
                
                if content:
                    texts.append(content)
                    metadatas.append({"title": title})
                    ids.append(str(uuid.uuid4()))
        
        # Add the documents and their metadata to the ChromaDB collection
        print(f"Adding {len(texts)} documents to the vector database...")
        if texts:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print("Knowledge base successfully loaded.")
        else:
            print("No valid documents found in the knowledge base file.")

    def query(self, text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the vector database for relevant documents based on a given text.
        Args:
            text (str): The text to query with (e.g., a user's question).
            n_results (int): The number of most similar documents to retrieve.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a retrieved document,
                                   its metadata, and its similarity score (distance).
        """
        # The ChromaDB query method finds the documents that are most semantically
        # similar to the input text's vector.
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results
        )

        # The results object is structured in a specific way, so we re-format it
        # to be more user-friendly.
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results

# --- Simple test block to demonstrate functionality ---
if __name__ == "__main__":
    # Path to the knowledge base file for testing
    # Note: In a real project, this path will be handled by main.py
    test_file_path = os.path.join("..", "..", "data", "knowledge_base.txt")

    print("--- Testing VectorDB Initialization and Loading ---")
    try:
        # Create an instance of the VectorDB
        test_db = VectorDB(test_file_path)
        
        # Load the knowledge base. This will create the embeddings and store them.
        test_db.load_knowledge_base()

        print("\n--- Testing a Query ---")
        # Now, let's search for a document related to a question.
        query_text = "What is the importance of a balanced diet for health?"
        results = test_db.query(query_text, n_results=2)
        
        print(f"Query: '{query_text}'\n")
        print("Retrieved Documents:")
        for result in results:
            # Print the title, similarity score, and a snippet of the content
            print(f"- Title: {result['metadata'].get('title', 'N/A')}")
            print(f"  Similarity Score (Lower is better): {result['distance']:.4f}")
            print(f"  Content Snippet: {result['content'][:100]}...\n")
            
    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        print("Please ensure you have all dependencies installed (chromadb, sentence-transformers) and the 'knowledge_base.txt' file exists at the correct path.")
