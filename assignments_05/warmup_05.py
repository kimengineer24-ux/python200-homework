import json
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# --- Completions API ---

# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)
print("\nAPI Q1:", response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
for temperature in [0, 0.7, 1.5]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    print(f"\nTemperature {temperature}:", response.choices[0].message.content)

# Temperature 0 is generally the most consistent and reproducible.
# Higher temperatures usually produce more varied and creative responses.

# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
for i, choice in enumerate(response.choices, start=1):
    print(f"\nPanda fact {i}:", choice.message.content)

# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work in detail."}],
    max_tokens=15
)
print("\nAPI Q4:", response.choices[0].message.content)

# The response is cut short because max_tokens limits output length.
# This helps control costs and response length in real applications.


# --- System Messages and Personas ---

# System Q1
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. Explain things simply and end with encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\nPatient tutor:", response.choices[0].message.content)

messages[0]["content"] = "You are a sarcastic pirate who explains programming using pirate language."
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\nPirate tutor:", response.choices[0].message.content)

# Changing the system message changes the model's tone and personality.

# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\nConversation memory:", response.choices[0].message.content)

# The model knows Jordan's name because the conversation history was
# included in the messages sent with this API call.


# --- Prompt Engineering ---

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

# Prompt Q1 — Zero-Shot
prompt = "Classify each review as positive, negative, or mixed. Give a numbered result for each:\n"
prompt += "\n".join(f"{i + 1}. {review}" for i, review in enumerate(reviews))
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("\nZero-shot results:", response.choices[0].message.content)

# Prompt Q2 — One-Shot
prompt = """
Classify each review as positive, negative, or mixed.
Follow this example:

Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify these reviews:
"""
prompt += "\n".join(f"{i + 1}. {review}" for i, review in enumerate(reviews))
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("\nOne-shot results:", response.choices[0].message.content)

# One example helps demonstrate the expected format and classification.

# Prompt Q3 — Few-Shot
prompt = """
Classify each review as positive, negative, or mixed.

Examples:
Review: "Excellent service and a friendly team."
Sentiment: positive

Review: "The product broke and customer service ignored me."
Sentiment: negative

Review: "The price was good, but delivery was slow."
Sentiment: mixed

Now classify these reviews:
"""
prompt += "\n".join(f"{i + 1}. {review}" for i, review in enumerate(reviews))
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("\nFew-shot results:", response.choices[0].message.content)

# Zero-shot is useful for simple tasks, one-shot demonstrates a format,
# and few-shot is useful when more examples are needed for consistency.

# Prompt Q4 — Calculation
prompt = """
A data engineer earns $85,000 per year. She gets a 12% raise,
then takes a new job that pays $7,500 more per year than her
post-raise salary. Show a brief calculation and clearly label
the final annual salary.
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("\nSalary calculation:", response.choices[0].message.content)

# A clear calculation prompt helps the model organize the steps and avoid
# overlooking one of the two salary increases.

# Prompt Q5 — Structured Output
review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)
prompt = f"""
Analyze this review and return ONLY valid JSON with these keys:
sentiment, confidence (a float from 0 to 1), and reason (one sentence).

Review: {review}
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
raw_response = response.choices[0].message.content
print("\nRaw JSON response:", raw_response)

try:
    result = json.loads(raw_response)
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except (json.JSONDecodeError, TypeError, KeyError) as error:
    print("Could not parse JSON:", error)
    print("Raw response for debugging:", raw_response)

# Prompt Q6 — Delimiters
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("\nPasta steps:", response.choices[0].message.content)

# Test with regular prose that does not contain instructions.
plain_text = "The morning sun warmed the quiet garden. Birds gathered in the trees."
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{plain_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("\nNon-instruction text:", response.choices[0].message.content)

# Delimiters separate user-provided text from instructions and help prevent
# the model from confusing the two.


# --- Local Models with Ollama ---

ollama_prompt = "Explain what a large language model is in two sentences."

try:
    result = subprocess.run(
        ["ollama", "run", "qwen3:0.6b", ollama_prompt],
        capture_output=True,
        text=True,
        timeout=180
    )
    print("\nOllama response:", result.stdout)
    if result.stderr:
        print("Ollama notes:", result.stderr)
except (FileNotFoundError, subprocess.TimeoutExpired) as error:
    print("\nOllama could not run:", error)
# Compare the same prompt using the OpenAI API.
openai_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": ollama_prompt}
    ],
    temperature=0.7,
    max_completion_tokens=150
)

print("\nOpenAI response:", openai_response.choices[0].message.content)

# Ollama output will be added here after the local model runs.
# Local models offer more privacy and offline access, but may have
# fewer capabilities and require local computing resources.# Ollama Exercise:
# Ollama was not available in this Codespace.
# The terminal returned: "ollama: command not found."
# No local model output was generated.

# Advantage of a local model: It can run without sending
# prompts to an external service.
# Disadvantage: It requires installation and sufficient
# local computer resources.