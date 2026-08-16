import os  # read environment variables
from pathlib import Path  # locate the project root .env file
from dotenv import load_dotenv  # load variables from .env into os.environ
from openai import OpenAI  # OpenAI-compatible client (also works with Groq)

# load .env from the project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Groq API key for LLM access

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not configured in the .env file")  # fail fast if missing


# OpenAI client pointed at Groq's API endpoint
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",  # Groq uses OpenAI-compatible API
)

MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")  # model name from .env
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))  # max response tokens from .env


def generate_answer(prompt: str) -> str:
    """Send a prompt to the LLM and return the generated answer text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",  # we send everything as a user message
                    "content": prompt,
                }
            ],
            temperature=0.2,  # low temperature = more deterministic, factual answers
            max_tokens=LLM_MAX_TOKENS,  # limit response length to control cost and latency
        )

        answer = response.choices[0].message.content  # extract the text from the response

        if not answer:
            raise ValueError("The LLM returned an empty response")  # guard against empty answers

        return answer.strip()  # clean up whitespace

    except Exception as error:
        raise RuntimeError(f"LLM request failed: {error}") from error  # wrap with context
