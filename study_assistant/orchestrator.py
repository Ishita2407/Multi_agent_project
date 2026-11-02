
"""
Orchestrator: runs agents in dependency order, routes review issues
back to the owning agent, repeats until approved (or max rounds),
and records an activity log.

LLM provider: Groq only.
"""

from __future__ import annotations

from datetime import datetime

from .agents import (
    FeedbackAgent,
    QuizAgent,
    ReviewerAgent,
    SummarizerAgent
)
from .llm import get_llm


class ActivityLog:

    def __init__(self):
        self.entries: list[dict] = []

    def add(self, agent: str, message: str):
        self.entries.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "agent": agent,
            "message": message
        })

    def lines(self) -> list[str]:
        return [
            f"[{e['time']}] {e['agent']:<13} {e['message']}"
            for e in self.entries
        ]

    def __str__(self):
        return "\n".join(self.lines())


class StudyAssistant:

    def __init__(
        self,
        llm=None,
        summarizer=None,
        quiz_agent=None,
        reviewer=None,
        feedback_agent=None,
        max_rounds: int = 3
    ):

        # Initialize Groq LLM unless an LLM instance is injected.
        self.llm = llm if llm is not None else get_llm()

        # Initialize agents.
        self.summarizer = summarizer or SummarizerAgent(self.llm)
        self.quiz_agent = quiz_agent or QuizAgent(self.llm)
        self.reviewer = reviewer or ReviewerAgent(self.llm)
        self.feedback_agent = feedback_agent or FeedbackAgent(self.llm)

        self.max_rounds = max_rounds

        self.log = ActivityLog()
        self.document = ""
        self.summary: dict = {}
        self.quiz: list[dict] = []
        self.unresolved: list[dict] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flush_notes(self, agent):
        for note in agent.pop_notes():
            self.log.add(agent.name, f"NOTE: {note}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def prepare(self, document: str, topic: str | None = None, on_update=None) -> dict:

        self.log = ActivityLog()
        self.unresolved = []

        L = self.log.add
        self.document = document

        L(
            "Orchestrator",
            f"Received document ({len(document.split())} words). "
            f"Engine: {self.llm.label}"
        )

        # 1. Summarization
        L("Summarizer", "Reading document and writing summary")

        self.summary = self.summarizer.run(document, topic)

        self._flush_notes(self.summarizer)
        if on_update:
            on_update("summary", self.summary)

        L(
            "Summarizer",
            f"Produced summary: {len(self.summary['key_points'])} "
            f"key points, {len(self.summary['key_terms'])} key terms "
            "-> passing to QuizCreator"
        )

        # 2. Quiz generation
        L(
            "QuizCreator",
            "Received document + summary; writing 5 multiple-choice questions"
        )

        self.quiz = self.quiz_agent.run(document, self.summary)
        self._flush_notes(self.quiz_agent)

        if on_update:
            on_update("quiz", self.quiz)

        L(
            "QuizCreator",
            f"Produced {len(self.quiz)} questions "
            "-> passing summary + quiz to Reviewer"
        )

        # 3. Review and correction loop
        approved = False

        for rnd in range(1, self.max_rounds + 1):

            L(
                "Reviewer",
                f"Round {rnd}: checking summary and quiz against the document"
            )

            report = self.reviewer.review(
                document,
                self.summary,
                self.quiz
            )

            self._flush_notes(self.reviewer)

            issues = report["issues"]

            if report["approved"]:

                L(
                    "Reviewer",
                    f"Round {rnd}: APPROVED, no issues found"
                )

                approved = True
                break

            L(
                "Reviewer",
                f"Round {rnd}: found {len(issues)} issue(s)"
            )

            for it in issues:

                where = it["target"] + (
                    f" #{it['index'] + 1}"
                    if it["index"] is not None
                    else ""
                )

                L(
                    "Reviewer",
                    f"  - {where}: {it['problem']}"
                )

            # Stop if maximum review rounds are reached.
            if rnd == self.max_rounds:

                L(
                    "Orchestrator",
                    "Max review rounds reached; "
                    "showing remaining issues to the learner"
                )

                self.unresolved = issues
                break

            # Route issues to their respective agents.
            s_issues = [
                i for i in issues
                if i["target"] == "summary"
            ]

            q_issues = [
                i for i in issues
                if i["target"] == "quiz"
            ]

            # Summary correction.
            if s_issues:

                L(
                    "Reviewer",
                    f"Requesting corrections from Summarizer "
                    f"({len(s_issues)} issue(s))"
                )

                self.summary = self.summarizer.revise(
                    document,
                    self.summary,
                    s_issues
                )
                if on_update:
                    on_update("summary", self.summary)

                self._flush_notes(self.summarizer)

                L(
                    "Summarizer",
                    "Revised summary submitted for re-review"
                )

            # Quiz correction.
            if q_issues:

                L(
                    "Reviewer",
                    f"Requesting corrections from QuizCreator "
                    f"({len(q_issues)} issue(s))"
                )

                self.quiz = self.quiz_agent.revise(
                    document,
                    self.summary,
                    self.quiz,
                    q_issues
                )
                if on_update:
                    on_update("quiz", self.quiz)

                self._flush_notes(self.quiz_agent)

                L(
                    "QuizCreator",
                    "Revised quiz submitted for re-review"
                )

        if approved:
            self.unresolved = []

        L(
            "Orchestrator",
            "Study material ready"
            + (
                ""
                if approved
                else " (with unresolved review issues)"
            )
        )
        # Notify Streamlit that the review process is complete.
        if on_update:
            on_update(
                "review_complete",
                {
                    "approved": approved,
                    "unresolved": self.unresolved
                }
            )
        return {
            "summary": self.summary,
            "quiz": self.quiz,
            "approved": approved,
            "unresolved": self.unresolved,
            "log": self.log.lines()
        }

    # ------------------------------------------------------------------
    # Learner evaluation
    # ------------------------------------------------------------------

    def evaluate(self, answers: list[int | None]) -> dict:

        self.log.add(
            "Feedback",
            f"Received learner answers "
            f"({sum(a is not None for a in answers)}"
            f"/{len(self.quiz)} answered); "
            "scoring and writing feedback"
        )

        result = self.feedback_agent.run(
            self.document,
            self.quiz,
            answers
        )

        self._flush_notes(self.feedback_agent)

        self.log.add(
            "Feedback",
            f"Score {result['score']}/{result['total']}; "
            "feedback and study plan ready"
        )

        return result

    @staticmethod
    def answer_key(quiz: list[dict]) -> list[str]:
        return [
            f"Q{i + 1}: {'ABCD'[q['answer_index']]}. "
            f"{q['options'][q['answer_index']]}"
            for i, q in enumerate(quiz)
        ]
