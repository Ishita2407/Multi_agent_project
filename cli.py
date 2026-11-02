"""Command-line runner.

  python cli.py sample_docs/generative_ai.txt                 # full report (summary, quiz, key, log)
  python cli.py sample_docs/generative_ai.txt --take-quiz     # interactive: answer, then get feedback
  python cli.py sample_docs/generative_ai.txt --answers ABCDA # non-interactive answers
"""
import argparse
import sys

from study_assistant import StudyAssistant, load_document
from study_assistant.document import DocumentError
from study_assistant.llm import LLMError, get_llm

L = "ABCD"


def print_summary(s):
    print("\n=== SUMMARY ===")
    print(s["overview"])
    for i, p in enumerate(s["key_points"], 1):
        print(f"  {i}. {p}")
    print("Key terms:", ", ".join(s["key_terms"]))


def print_quiz(quiz):
    print("\n=== QUIZ ===")
    for i, q in enumerate(quiz, 1):
        print(f"\nQ{i}. {q['question']}")
        for j, o in enumerate(q["options"]):
            print(f"   {L[j]}. {o}")


def print_feedback(fb):
    print(f"\n=== YOUR RESULT: {fb['score']}/{fb['total']} ({fb['percent']}%) ===")
    print(fb["overall"])
    for i, d in enumerate(fb["details"], 1):
        mark = "OK " if d["correct"] else "X  "
        print(f"\n{mark}Q{i}: you answered '{d['your_answer']}'  | correct: '{d['correct_answer']}'")
        print(f"    {d['feedback']}")
    print("\nStudy plan:")
    for step in fb["study_plan"]:
        print(f"  - {step}")


def ask_answers(n):
    answers = []
    for i in range(n):
        while True:
            a = input(f"Answer for Q{i + 1} (A-D, Enter to skip): ").strip().upper()
            if a == "" or a in L:
                answers.append(L.index(a) if a else None)
                break
    return answers


def main():
    ap = argparse.ArgumentParser(description="Multi-Agent AI Study Assistant")
    ap.add_argument("document", help="path to a .pdf or .txt file")
    ap.add_argument("--topic", help="optional focus, e.g. 'Generative AI'")
    ap.add_argument("--take-quiz", action="store_true", help="answer the quiz interactively")
    ap.add_argument("--answers", help="answers as letters, e.g. ABCDA (skips the prompts)")
    ap.add_argument("--provider", help="ollama | groq | gemini | openrouter | custom | offline "
                                       "(default: auto-detect, see README)")
    args = ap.parse_args()

    try:
        text = load_document(args.document)
    except DocumentError as e:
        sys.exit(f"Error: {e}")

    try:
        llm = get_llm(args.provider)          # None -> offline mode
    except LLMError as e:
        sys.exit(f"Error: {e}")
    app = StudyAssistant(llm=llm)
    res = app.prepare(text, args.topic)
    print_summary(res["summary"])
    print_quiz(res["quiz"])

    answers = None
    if args.answers:
        answers = [L.index(c) if c in L else None for c in args.answers.upper()]
        answers += [None] * (len(res["quiz"]) - len(answers))
    elif args.take_quiz:
        print()
        answers = ask_answers(len(res["quiz"]))

    print("\n=== ANSWER KEY ===")
    print("\n".join(app.answer_key(res["quiz"])))
    if res["unresolved"]:
        print("\nWARNING: unresolved review issues:", *[f"- {i['problem']}" for i in res["unresolved"]], sep="\n  ")
    if answers is not None:
        print_feedback(app.evaluate(answers))

    print("\n=== AGENT ACTIVITY LOG ===")
    print(app.log)


if __name__ == "__main__":
    main()
