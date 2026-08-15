from prompts import build_prompt
from client import generate_answer


# Dummy data from Member 5
question = "What was the company's revenue in 2025?"

context = """
In 2025, Contexta Technologies reported a total revenue
of $25 million.

Source: annual_report_2025.pdf
Page: 42
Section: Financial Results
"""


# Step 1: Build the prompt
prompt = build_prompt(question, context)

print("===== SENDING TO GROQ =====")
print(prompt)


# Step 2: Send the prompt to the LLM
answer = generate_answer(prompt)

print("\n===== GROQ ANSWER =====")
print(answer)