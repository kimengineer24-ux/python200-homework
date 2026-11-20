from dotenv import load_dotenv
import os
import string
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")# --- RAG Concepts ---
# Concepts Q1

# Scenario A: RAG
# RAG is best because the legal team needs answers from hundreds of
# documents that are updated regularly.

# Scenario B: Fine-tuning
# Fine-tuning will help the model learn and consistently use the
# company's specific writing style.

# Scenario C: Prompt Engineering
# Prompt engineering is enough because the analyst only needs help
# with one short report.# --- Concepts Q2 ---
# A confidently wrong answer is dangerous because people may trust it
# and act without checking. For example, someone could follow incorrect
# medical instructions and get hurt. Saying "I'm not sure" encourages
# people to verify the information instead of blindly trusting it.# --- Concepts Q3 ---
# 1. Extract text from source documents.
# 2. Split text into chunks.
# 3. Convert text chunks into embeddings.
# 4. Receive the user's query.
# 5. Embed the user's query.
# 6. Retrieve the most relevant chunks.
# 7. Inject retrieved chunks into the prompt.
# 8. Generate a response from the LLM.# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]# --- Keyword Question 1 ---

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

# hours.txt was selected because it contains the keyword "weekends,"
# which directly matches the question.# --- Keyword Question 2 ---

query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

# No document was selected because none of the documents contain
# the exact keywords. Semantic search could better understand
# related meanings, but the documents don't confirm caffeine-free options.# --- Keyword Question 3 ---

# Prediction: None found. The loyalty document discusses rewards,
# but it uses the word "loyalty" instead of "rewards."

query = "How do I sign up for rewards?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

# My prediction was correct. Keyword retrieval missed the connection
# between rewards and the loyalty program because the wording differed.# --- Semantic RAG Concepts ---
# Semantic Question 1

# A vector embedding turns text into numbers that represent its meaning.
# A score of 0.85 is more relevant than 0.30 because the meanings are closer.
# Semantic search can find related information even when the exact words
# are different because it compares meaning, not just matching keywords.# --- Semantic Question 2 ---

# | Feature           | Keyword RAG              | Semantic RAG              |
# |-------------------|--------------------------|---------------------------|
# | What is compared? | Exact word overlap      | Meaning of text           |
# | What is retrieved?| Full document           | Relevant text chunks      |
# | Can it handle synonyms? | No                 | Yes                       |
# | Storage format    | Plain text dictionary   | Vector store / index      |
# | Relevance score   | Keyword overlap count   | Cosine similarity         |
# --- LlamaIndex ---
# LlamaIndex Question 1

docs_dir = "../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs"

docs = SimpleDirectoryReader(docs_dir).load_data()

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")
index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print("\nQuestion:", question)
    response = query_engine.query(question)
    print("Answer:", response)

    for node in response.source_nodes:
        print("Score:", node.score)
        print("Source:", node.node.text[:150])
# Employee benefits: I checked whether the retrieved text directly addressed the question.
# The answer sounded confident, but I compared it with the source to check accuracy.
# Security policies: I checked whether the retrieved text addressed the actual policies.
# I was surprised that related information can be retrieved without fully answering a question.
# LlamaIndex Question 2

question = "What employee benefits does BrightLeaf offer?"

for k in [1, 5]:
    print(f"\n--- Testing similarity_top_k={k} ---")

    engine = index.as_query_engine(similarity_top_k=k)
    response = engine.query(question)

    print("Answer:", response)

    for node in response.source_nodes:
        print("Score:", node.score)# With more retrieved chunks, the answer may include more detail.
# However, more information is not always better because irrelevant
# chunks can distract from the question.# LlamaIndex Question 3

question = "What is BrightLeaf's office lunch menu?"

response = query_engine.query(question)

print("\nQuestion:", question)
print("Answer:", response)

for node in response.source_nodes:
    print("\nDocument:", node.node.metadata.get("file_name"))
    print("Score:", node.score)
    print("Retrieved text:", node.node.text)# I chose this question because the information may not
# be included in the BrightLeaf documents. I want to see
# whether the AI recognizes missing information or guesses.# --- LlamaIndex Question 4 ---

llm = OpenAI(model="gpt-4o-mini")

faithfulness = FaithfulnessEvaluator(llm=llm)
relevancy = RelevancyEvaluator(llm=llm)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What is the CEO's favorite color?"
]

for q in questions:
    print("\nQuestion:", q)

    response = query_engine.query(q)
    print("Answer:", response)

    faith_result = faithfulness.evaluate_response(response=response)
    rel_result = relevancy.evaluate_response(
        query=q,
        response=response
    )

    print("Faithfulness score:", faith_result.score)
    print("Relevancy score:", rel_result.score)# A faithfulness score of 1.0 means the answer is supported
# by the retrieved information. A 0.0 means it is unsupported.
#
# Relevancy measures how well the answer addresses the question.
# Faithfulness checks whether the answer is supported by the sources.
#
# The scores may change depending on how much relevant information
# the documents contain.
#
# LLM-as-a-judge uses another AI model to evaluate responses.
# It helps assess answer quality beyond simple right-or-wrong checks.