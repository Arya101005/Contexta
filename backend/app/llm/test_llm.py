from prompts import build_prompt
from client import generate_answer
from guardrails import validate_answer


# Dummy data from Member 5
question = "What was the company's revenue in 2025?"

context = """
In 2025, Contexta Technologies reported a total revenue
of $25 million.

Source: annual_report_2025.pdf
Page: 42
Section: Financial Results
"""


# 1. Build the prompt
prompt = build_prompt(question, context)

print("===== STEP 1: PROMPT CREATED =====")
print(prompt)


# 2. Send the prompt to Groq
answer = generate_answer(prompt)

print("\n===== STEP 2: GROQ ANSWER =====")
print(answer)


# 3. Validate the answer
validated_answer = validate_answer(answer, context)

print("\n===== STEP 3: FINAL VALIDATED ANSWER =====")
print(validated_answer)