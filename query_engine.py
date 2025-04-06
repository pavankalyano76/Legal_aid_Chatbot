import os
import argparse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.llms.ollama import Ollama

from llama_index.core import StorageContext, load_index_from_storage
from sentence_transformers import SentenceTransformer


# Define the argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Legal Advisor CLI")

    # Subcommands
    subparsers = parser.add_subparsers(dest='command')

    # Ingestion command
    ingestion_parser = subparsers.add_parser('ingest', help="Ingest data into the index")
    ingestion_parser.add_argument('data_path', type=str, help="Path to the data to ingest")

    # Query command
    query_parser = subparsers.add_parser('query', help="Query the RAG model")
    query_parser.add_argument('query_text', type=str, help="Query text for the RAG model")

    # Chat command (interactive mode)
    subparsers.add_parser('chat', help="Interactive chat with the AI model")

    return parser.parse_args()

# Set up necessary models and directories
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Ollama(model="llama3.2:latest", request_timeout=120.0)

persist_dir = "C:\\Users\\kavet\\Downloads\\GNU\\persisted_index"
os.makedirs(persist_dir, exist_ok=True)

storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

# Define the custom prompt template
template = (
    "Given the context information and not prior knowledge, "
    "You are a Legal Aid Assistant designed to support users with questions about disability rights and protections. "
    "Provide clear, concise legal guidance based on disability laws, benefits, and related government services. "
    "Reference relevant acts or policy sections when possible, and explain in a way that is easy to understand. "
    "If applicable, give a simple real-life example to help clarify the legal point.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = PromptTemplate(template)

# Make sure RAGQueryEngine class is defined before you use it
class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine for custom retrieval and response synthesis."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        
        


        context_str = "\n\n".join([node.get_content() for node in nodes])
        formatted_prompt = qa_template.format(context_str=context_str, query_str=query_str)

        
        # Generate response
        response_obj = self.response_synthesizer.synthesize(query=formatted_prompt, nodes=nodes)
        return response_obj
    
    
    


# Function to handle the interactive chat
def chat():
    print("Welcome to Legal Advisor AI! Type 'exit' to quit the chat.")
    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Query the RAG model and return a response
        print(f"Running query: {user_input}")
        
        # Load index
        try:
            index = load_index_from_storage(storage_context)
            print("Index loaded successfully.")
        except Exception as e:
            print(f"Error loading index: {e}")
            continue

        retriever = index.as_retriever()
        synthesizer = get_response_synthesizer(response_mode="compact")
        query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
        
        response = query_engine.query(user_input)
        print("Assistant:", response)

# Simulate argparse in notebook

# Choose which mode to run
mode = 'chat'  # Options: 'ingest', 'query', 'chat'

# Provide required inputs for each mode
#data_path = 'C:\\Users\\kavet\\Downloads\\GNUHack\\Data_Folder'
query_text = "Where can I find estimates about the employment situation for individuals with disabilities?"


chat()
