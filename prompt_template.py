from langchain_core.prompts import PromptTemplate


mcq_prompt = PromptTemplate(
    input_variables=["context", "topic", "difficulty", "mcq_count"],
    template="""
You are a subject expert and a very good teacher.

Create exactly {mcq_count} {difficulty} multiple-choice questions about "{topic}" using ONLY the context.

Rules:
- Each question must have 4 choices.
- Only one choice is correct.
- Answer text must exactly match one choice.
- Add a short explanation.
- Return ONLY JSON list.

Context:
{context}

Output format:
[
  {{
    "question": "...",
    "choices": ["...", "...", "...", "..."],
    "answer": "...",
    "explanation": "..."
  }}
]
""",
)

notes_prompt = PromptTemplate(
    input_variables=["context", "topic", "difficulty"],
    template="""
You are a subject expert and a very good teacher.

Write short notes for "{topic}" at {difficulty} level using ONLY the context.
Keep it clear and concise with headings and bullet points.

Context:
{context}
""",
)

flashcards_prompt = PromptTemplate(
    input_variables=["context", "topic", "difficulty"],
    template="""
You are a subject expert and a very good teacher.

Create 10 flashcards for "{topic}" at {difficulty} level using ONLY the context.
Format:
Q: ...
A: ...

Context:
{context}
""",
)

viva_prompt = PromptTemplate(
    input_variables=["context", "topic", "difficulty"],
    template="""
You are a strict examiner.

Create 10 viva questions for "{topic}" at {difficulty} level using ONLY the context.
For each question, add a short model answer.

Return in this exact format only:
Q1: ...
A1: ...
Q2: ...
A2: ...

Context:
{context}
""",
)


def get_prompt(task: str):
    prompts = {
        "MCQ": mcq_prompt,
        "Notes": notes_prompt,
        "Flashcards": flashcards_prompt,
        "Viva": viva_prompt,
    }
    return prompts[task]
