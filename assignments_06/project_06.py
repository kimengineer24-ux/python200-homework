import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
# Load API key
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: Check your .env file.")

# Check document folder
docs_dir = Path("../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs")

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Load documents
documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"Number of documents loaded: {len(documents)}")

for doc in documents:
    print("File:", doc.metadata["file_name"])
# Build the index
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4.1-mini")
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")
# Step 4: Query the Assistant
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\nQuestion:", question)

    response = query_engine.query(question)
    print("Answer:", response)

    for node in response.source_nodes[:1]:
        print("Document:", node.node.metadata.get("file_name"))
        print("Similarity score:", node.score)
        print("Retrieved text:", node.node.get_content()[:200])    