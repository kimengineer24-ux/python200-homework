import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# --- Task 1: Setup and System Prompt ---

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """
You are a professional job application coach helping career changers
prepare strong resumes and cover letters.

Stay focused on job application materials. Help users communicate
their real experience clearly and confidently. Never invent credentials,
skills, accomplishments, or work history.

Always remind users to review and edit your suggestions before
submitting them to an employer. You may not know the specific norms
of every industry, so users should apply their own judgment.

Be practical, encouraging, and specific.
"""

# I included instructions not to invent qualifications so the tool
# helps users present their real experience accurately.


# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.
Rewrite each resume bullet to be more specific, results-oriented,
and compelling. Use strong action verbs. Do not invent facts.

Return ONLY valid JSON. Each item must have:
"original" and "improved" keys.

Bullet points:--- BEGIN USER BULLET POINTS ---
{bullet_text}
--- END USER BULLET POINTS ---
"""

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)

    try:
        results = json.loads(response)
        for item in results:
            print("\nOriginal:", item["original"])
            print("Improved:", item["improved"])
        return results
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        print("Could not parse JSON:", error)
        print("Raw response:", response)
        return []

        # --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional cover letter coach. "
                "Write a tailored opening paragraph of 3-5 sentences. "
                "Use only the qualifications and experience provided. "
                "Never invent credentials or accomplishments."
            )
        },
        {
            "role": "user",
            "content": (
                "Job: Marketing Assistant. "
                "Background: Two years of experience creating social media content. "
                "Write a tailored cover letter opening."
            )
        },
        {
            "role": "assistant",
            "content": (
                "I am excited to apply for the Marketing Assistant position. "
                "My two years of experience creating social media content "
                "have helped me develop practical communication and content "
                "development skills."
            )
        },
        {
            "role": "user",
            "content": (
                "Job: Data Analyst. "
                "Background: Experience organizing spreadsheets and preparing reports. "
                "Write a tailored cover letter opening."
            )
        },
        {
            "role": "assistant",
            "content": (
                "I am interested in the Data Analyst position. "
                "My experience organizing spreadsheets and preparing reports "
                "has strengthened my attention to detail and ability to "
                "communicate information clearly."
            )
        },
        {
            "role": "user",
            "content": (
                f"Job title: {job_title}\n"
                f"Background: {background}\n"
                "Write a tailored 3-5 sentence cover letter opening."
            )
        }
    ]

    response = get_completion(messages)
    return response.strip()

    # --- Task 4: Content Moderation ---

def is_safe(text: str) -> bool:
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = response.results[0].flagged

    if flagged:
        print("This content was flagged. Please revise your message.")

    return not flagged

    # --- Task 5: Interactive Career Coach Chatbot ---

def run_chatbot():
    system_prompt = """
    You are a helpful, patient career coach for people changing careers.
    Help users improve resumes, prepare cover letters, and discuss job
    applications. Stay focused on career development. Do not invent
    qualifications or work experience. Remind users to review and edit
    suggestions before submitting them.
    """

    conversation = [
        {"role": "system", "content": system_prompt}
    ]

    print("\nWelcome to the AI Career Coach!")
    print("Type 'quit' or 'exit' to leave.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("Good luck with your career journey!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("Enter your resume bullet points one at a time.")
            print("Type DONE when finished.")
            bullets = []

            while True:
                bullet = input("Bullet: ").strip()
                if bullet.upper() == "DONE":
                    break
                if bullet and is_safe(bullet):
                    bullets.append(bullet)

            if bullets:
                rewrite_bullets(bullets)
            continue

        if "cover letter" in user_input.lower():
            job_title = input("What is the job title? ").strip()
            background = input("Describe your experience: ").strip()

            if not is_safe(job_title) or not is_safe(background):
                continue

            letter = generate_cover_letter(job_title, background)
            print("\nCover Letter:\n", letter)
            continue

        conversation.append({
            "role": "user",
            "content": user_input
        })

        reply = get_completion(conversation)
        print("\nCareer Coach:", reply)

        conversation.append({
            "role": "assistant",
            "content": reply
        })


if __name__ == "__main__":
    run_chatbot()
        