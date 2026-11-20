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

# Step 4 Reflection:
# The assistant sounded confident when the documents contained relevant information.
# When information was missing, it could not provide a supported answer.
# I was surprised that retrieval could return information that did not fully answer a question.

# Step 5: Find a Failure
hard_question = "What is Groundwork Coffee's exact annual revenue and profit margin?"

print("\nStep 5: Failure Test")
print("Question:", hard_question)

failure_response = query_engine.query(hard_question)

print("Full Response:", failure_response)

for i, node in enumerate(failure_response.source_nodes, start=1):
    print(f"\nSource {i}:")
    print("Document:", node.node.metadata.get("file_name"))
    print("Score:", node.score)
    print("Retrieved text:", node.node.get_content())

# The documents do not contain exact revenue or profit figures.
# The retriever may return unrelated information, showing how RAG
# can struggle when the source documents do not contain the answer.

# Step 6: Final Reflection
# LlamaIndex requires fewer lines than building a RAG pipeline manually
# because it handles document loading, embeddings, indexing, and retrieval.
# Another use case would be a school answering questions using its handbooks and policies.
# RAG can fail when documents are missing, outdated, or irrelevant to the question.
