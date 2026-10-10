import json
import os
import re
import urllib.request
from pathlib import Path

from checker import check_answer

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "neuron_docs.json"
QUESTIONS_FILE = ROOT / "questions.json"
LOG_FILE = ROOT / "data" / "attempt_log.jsonl"

BASE_URL = os.getenv(
    "NEURON_OPENAI_BASE_URL",
    "http://localhost:8000/v1"
).rstrip("/")

SOURCE_URL = (
    "https://awsdocs-neuron.readthedocs-hosted.com/"
    "en/latest/about-neuron/what-is-neuron.html"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(question, paragraphs, top_k=6):
    stop_words = {
        "what", "does", "the", "and", "for", "are", "how",
        "which", "with", "from", "this", "that", "does"
    }

    terms = {
        w for w in re.findall(r"[a-zA-Z0-9]+", question.lower())
        if len(w) > 2 and w not in stop_words
    }

    ranked = []
    for paragraph in paragraphs:
        words = set(re.findall(r"[a-zA-Z0-9]+", paragraph.lower()))
        overlap = len(terms & words)
        if overlap:
            ranked.append((overlap, paragraph))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [p for _, p in ranked[:top_k]]

    return selected or paragraphs[:top_k]


def api_request(path, payload=None):
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"}
    data = None

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url, data=data, headers=headers,
        method="POST" if data is not None else "GET"
    )

    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def get_model():
    result = api_request("/models")
    models = result.get("data", [])

    if not models:
        raise RuntimeError("No models returned by the model endpoint.")

    return models[0]["id"]


def ask_llm(question, context, feedback=""):
    model = get_model()

    system_prompt = (
        "You are a documentation question-answering assistant. "
        "Answer only using the supplied source context. "
        "Do not invent facts. If the context does not support a claim, "
        "say that the source does not specify it. "
        "Include the exact source URL in your answer."
    )

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCE CONTEXT:\n{context}\n\n"
        f"CHECKER FEEDBACK FROM THE PREVIOUS ATTEMPT:\n"
        f"{feedback or 'This is the first attempt.'}\n\n"
        "Provide a clear, concise, source-grounded answer. "
        "Address the checker feedback if this is a retry."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 500
    }

    result = api_request("/chat/completions", payload)
    return result["choices"][0]["message"]["content"].strip()


def run_question(question, docs, max_attempts=3, run_id=1):
    paragraphs = docs.get("paragraphs", [])
    relevant = retrieve(question["question"], paragraphs)

    context = "\n\n".join(relevant)
    feedback = ""
    attempts = []

    for attempt_no in range(1, max_attempts + 1):
        answer = ask_llm(
            question["question"],
            context,
            feedback
        )

        evaluation = check_answer(answer, question)

        record = {
            "run_id": run_id,
            "question_id": question["id"],
            "question": question["question"],
            "attempt": attempt_no,
            "answer": answer,
            "evaluation": evaluation,
            "source_url": docs.get("url", SOURCE_URL)
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        attempts.append(record)

        print(
            f"[{question['id']}] Attempt {attempt_no}: "
            f"score={evaluation['score']}, "
            f"passed={evaluation['passed']}"
        )

        if evaluation["passed"]:
            break

        feedback = evaluation["feedback"]

    final = attempts[-1]

    return {
        "run_id": run_id,
        "question_id": question["id"],
        "question": question["question"],
        "answer": final["answer"],
        "score": final["evaluation"]["score"],
        "passed": final["evaluation"]["passed"],
        "attempts_used": len(attempts),
        "missing_facts": final["evaluation"]["missing_facts"]
    }


def main():
    docs = load_json(DATA_FILE)
    questions = load_json(QUESTIONS_FILE)

    print("Model endpoint:", BASE_URL)
    print("Scraped source:", docs.get("url"))
    print("Questions:", len(questions))

    results = [
        run_question(q, docs, max_attempts=3)
        for q in questions
    ]

    print("\nFinal results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
