from prompts import build_prompt


question = "What was the company's revenue in 2025?"

context = """
In 2025, Contexta Technologies reported a total revenue
of $25 million.

Source: annual_report_2025.pdf
Page: 42
Section: Financial Results
"""


final_prompt = build_prompt(question, context)

print("===== FINAL PROMPT =====")
print(final_prompt)