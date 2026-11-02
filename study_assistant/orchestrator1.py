
"""
Orchestrator: coordinates the study-assistant agents.

Pipeline:
    1. Generate summary.
    2. Generate quiz.
    3. Return the material immediately.
    4. Review and revise the material separately.
    5. Evaluate learner answers against the exact quiz version attempted.

LLM provider: Groq only.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from threading import Lock

from .agents import (
    FeedbackAgent,
    QuizAgent,
    ReviewerAgent,
    SummarizerAgent,
)
from .llm import get_llm


class ActivityLog:

    def __init__(self):
        self.entries: list[dict] = []
        self._lock = Lock()

    def add(self, agent: str, message: str):
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "agent": agent,
            "message": message,
        }

        with self._lock:
            self.entries.append(entry)

    def lines(self) -> list[str]:
        with self._lock:
            entries = deepcopy(self.entries)

        return [
            f"[{e['time']}] {e['agent']:<13} {e['message']}"
            for e in entries
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
        max_rounds: int = 3,
    ):

        self.llm = llm if llm is not None else get_llm()

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

    def _record_material(self, summary: dict, quiz: list[dict]):
        """Store independent copies of the current material."""

        self.summary = deepcopy(summary)
        self.quiz = deepcopy(quiz)

    # ------------------------------------------------------------------
    # Stage 1: Generate summary and quiz
    # ------------------------------------------------------------------

    def generate_material(
        self,
        document: str,
        topic: str | None = None,
        on_update=None,
    ) -> dict:
        """
        Generate the summary and quiz without waiting for Reviewer.

        The method returns as soon as Quiz Creator finishes. The caller
        can then make the quiz available while review runs separately.
        """

        self.log = ActivityLog()
        self.unresolved = []
        self.document = document

        self.log.add(
            "Orchestrator",
            f"Received document ({len(document.split())} words). "
            f"Engine: {self.llm.label}",
        )

        # 1. Generate summary.
        self.log.add(
            "Summarizer",
            "Reading document and writing summary",
        )

        summary = self.summarizer.run(document, topic)
        self._flush_notes(self.summarizer)

        self.summary = deepcopy(summary)

        if on_update:
            on_update("summary", deepcopy(summary))

        self.log.add(
            "Summarizer",
            f"Produced summary: {len(summary['key_points'])} "
            f"key points, {len(summary['key_terms'])} key terms "
            "-> passing to QuizCreator",
        )

        # 2. Generate quiz.
        self.log.add(
            "QuizCreator",
            "Received document + summary; writing "
            "multiple-choice questions",
        )

        quiz = self.quiz_agent.run(document, summary)
        self._flush_notes(self.quiz_agent)

        self.quiz = deepcopy(quiz)

        if on_update:
            on_update("quiz", deepcopy(quiz))

        self.log.add(
            "QuizCreator",
            f"Produced {len(quiz)} questions "
            "-> material ready for learner",
        )

        return {
            "summary": deepcopy(summary),
            "quiz": deepcopy(quiz),
            "approved": None,
            "review_complete": False,
            "unresolved": [],
            "log": self.log.lines(),
        }

    # ------------------------------------------------------------------
    # Stage 2: Review and correction
    # ------------------------------------------------------------------

    def review_material(
        self,
        summary: dict | None = None,
        quiz: list[dict] | None = None,
        on_update=None,
    ) -> dict:
        """
        Review and revise generated material.

        This method is designed to be called from a background worker.
        It does not call Streamlit APIs.

        The caller should pass the initial summary and quiz returned by
        generate_material(). The method works on independent copies.
        """

        if not self.document:
            raise ValueError(
                "No document is available. Generate study material first."
            )

        # Work on independent copies. This prevents the reviewer from
        # modifying the quiz currently displayed to the learner.
        reviewed_summary = deepcopy(
            summary if summary is not None else self.summary
        )
        reviewed_quiz = deepcopy(
            quiz if quiz is not None else self.quiz
        )

        unresolved = []
        approved = False

        self.log.add(
            "Reviewer",
            "Background review started",
        )

        if on_update:
            on_update(
                "review_status",
                {"message": "Reviewer is checking the study material"},
            )

        for rnd in range(1, self.max_rounds + 1):

            self.log.add(
                "Reviewer",
                f"Round {rnd}: checking summary and quiz",
            )

            if on_update:
                on_update(
                    "review_status",
                    {
                        "message": f"Reviewer is checking the material "
                                   f"(round {rnd}/{self.max_rounds})"
                    },
                )

            report = self.reviewer.review(
                self.document,
                reviewed_summary,
                reviewed_quiz,
            )

            self._flush_notes(self.reviewer)

            issues = report.get("issues", [])

            if report.get("approved", False):

                self.log.add(
                    "Reviewer",
                    f"Round {rnd}: approved",
                )

                approved = True
                unresolved = []
                break

            self.log.add(
                "Reviewer",
                f"Round {rnd}: found {len(issues)} issue(s)",
            )

            if rnd == self.max_rounds:
                unresolved = deepcopy(issues)

                self.log.add(
                    "Orchestrator",
                    "Maximum review rounds reached",
                )
                break

            summary_issues = [
                issue for issue in issues
                if issue.get("target") == "summary"
            ]

            quiz_issues = [
                issue for issue in issues
                if issue.get("target") == "quiz"
            ]

            # Revise summary if required.
            if summary_issues:

                self.log.add(
                    "Reviewer",
                    f"Requesting summary corrections "
                    f"({len(summary_issues)} issue(s))",
                )

                reviewed_summary = self.summarizer.revise(
                    self.document,
                    reviewed_summary,
                    summary_issues,
                )

                self._flush_notes(self.summarizer)

                self.log.add(
                    "Summarizer",
                    "Revised summary submitted for re-review",
                )

            # Revise quiz if required.
            if quiz_issues:

                self.log.add(
                    "Reviewer",
                    f"Requesting quiz corrections "
                    f"({len(quiz_issues)} issue(s))",
                )

                reviewed_quiz = self.quiz_agent.revise(
                    self.document,
                    reviewed_summary,
                    reviewed_quiz,
                    quiz_issues,
                )

                self._flush_notes(self.quiz_agent)

                self.log.add(
                    "QuizCreator",
                    "Revised quiz submitted for re-review",
                )

        if approved:
            unresolved = []

        # Store the reviewed version as the latest version.
        self._record_material(
            reviewed_summary,
            reviewed_quiz,
        )

        self.unresolved = deepcopy(unresolved)

        result = {
            "summary": deepcopy(reviewed_summary),
            "quiz": deepcopy(reviewed_quiz),
            "approved": approved,
            "review_complete": True,
            "unresolved": deepcopy(unresolved),
            "log": self.log.lines(),
        }

        self.log.add(
            "Orchestrator",
            "Background review completed",
        )

        result["log"] = self.log.lines()

        if on_update:
            on_update(
                "review_complete",
                {
                    "approved": approved,
                    "unresolved": deepcopy(unresolved),
                    "summary": deepcopy(reviewed_summary),
                    "quiz": deepcopy(reviewed_quiz),
                },
            )

        return result

    # ------------------------------------------------------------------
    # Backward-compatible pipeline
    # ------------------------------------------------------------------

    def prepare(
        self,
        document: str,
        topic: str | None = None,
        on_update=None,
    ) -> dict:
        """
        Run generation followed by review synchronously.

        Retained for compatibility with existing code and tests.
        The updated Streamlit app will use generate_material() and
        review_material() separately.
        """

        initial = self.generate_material(
            document,
            topic,
            on_update=on_update,
        )

        reviewed = self.review_material(
            initial["summary"],
            initial["quiz"],
            on_update=on_update,
        )

        return reviewed

    # ------------------------------------------------------------------
    # Learner evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        answers: list[int | None],
        quiz: list[dict] | None = None,
    ) -> dict:
        """
        Evaluate answers against the exact quiz version attempted.

        Passing quiz explicitly prevents background review revisions
        from changing the correct answers for an ongoing attempt.
        """

        quiz_to_grade = deepcopy(
            quiz if quiz is not None else self.quiz
        )

        if not quiz_to_grade:
            raise ValueError(
                "No quiz is available for evaluation."
            )

        self.log.add(
            "Feedback",
            f"Received learner answers "
            f"({sum(a is not None for a in answers)}"
            f"/{len(quiz_to_grade)} answered); "
            "scoring and writing feedback",
        )

        result = self.feedback_agent.run(
            self.document,
            quiz_to_grade,
            answers,
        )

        self._flush_notes(self.feedback_agent)

        self.log.add(
            "Feedback",
            f"Score {result['score']}/{result['total']}; "
            "feedback and study plan ready",
        )

        return result

    # ------------------------------------------------------------------
    # Answer key
    # ------------------------------------------------------------------

    @staticmethod
    def answer_key(quiz: list[dict]) -> list[str]:

        return [
            f"Q{i + 1}: {'ABCD'[q['answer_index']]}. "
            f"{q['options'][q['answer_index']]}"
            for i, q in enumerate(quiz)
        ]
