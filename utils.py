import json
import re


def parse_mcq_response(raw_text: str):
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            raise ValueError("Model response is not valid JSON.")
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Response must be a JSON list.")

    required = {"question", "choices", "answer", "explanation"}

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {i} is invalid.")
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Question {i} missing fields: {', '.join(sorted(missing))}")
        if not isinstance(item["choices"], list) or len(item["choices"]) != 4:
            raise ValueError(f"Question {i} must have 4 choices.")

    return data


def score_exam(exam, selected_answers):
    score = 0
    results = []

    for i, q in enumerate(exam):
        user_answer = selected_answers.get(i)
        correct_answer = q["answer"]

        if str(user_answer).strip() == str(correct_answer).strip():
            score += 1

        results.append(
            {
                "question": q["question"],
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "explanation": q["explanation"],
            }
        )

    return score, len(exam), results