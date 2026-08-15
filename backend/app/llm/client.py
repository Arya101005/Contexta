import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Load the .env file from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not configured in the .env file")


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


MODEL_NAME = "llama-3.3-70b-versatile"


def generate_answer(prompt: str) -> str:
    """
    Send a prompt to the Groq-hosted LLM and return the generated answer.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=500,
        )

        answer = response.choices[0].message.content

        if not answer:
            raise ValueError("The LLM returned an empty response")

        return answer.strip()

    except Exception as error:
        raise RuntimeError(f"LLM request failed: {error}") from error