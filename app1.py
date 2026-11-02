
import queue
import threading

import streamlit as st

from study_assistant.orchestrator1 import StudyAssistant
from study_assistant.document import DocumentError, load_document
from study_assistant.llm import LLMError

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Multi-Agent Study Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .landing-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 750;
        margin-bottom: 0.5rem;
    }

    .landing-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 2.5rem;
    }

    .landing-caption {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-top: 1rem;
    }

    div[data-testid="stFileUploader"] {
        width: 100%;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        min-height: 45px;
    }

    section[data-testid="stSidebar"] {
        min-width: 280px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# APPLICATION HEADING
# =========================================================

st.markdown(
    '<h1 class="landing-title">'
    '📚 Multi-Agent AI Study Assistant'
    '</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="landing-subtitle">'
    'Transform your documents into summaries, quizzes '
    'and personalized study plans.'
    '</p>',
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

DEFAULTS = {
    "active_section": "Summary",
    "result": None,
    "assistant": None,
    "feedback": None,
    "review_queue": None,
    "review_thread": None,
    "review_status": "not_started",
    "reviewed_result": None,
    "quiz_attempt": None,
    "quiz_submitted": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SUMMARY RENDERING
# =========================================================

def render_summary(summary):

    st.subheader("Document Summary")

    st.write(summary["overview"])

    st.divider()

    st.markdown("### Key Points")

    for point in summary["key_points"]:
        st.markdown(f"- {point}")

    st.divider()

    st.markdown("### Key Terms")

    st.write(", ".join(summary["key_terms"]))


# =========================================================
# REVIEWER BACKGROUND WORKER
# =========================================================

def reviewer_worker(
    assistant,
    summary,
    quiz,
    result_queue
):
    """
    Run Reviewer outside the Streamlit script thread.

    The worker communicates with Streamlit only through
    the thread-safe queue. It never accesses session state
    or calls Streamlit UI functions.
    """

    def send_update(stage, data):
        result_queue.put((stage, data))

    try:

        reviewed_result = assistant.review_material(
            summary=summary,
            quiz=quiz,
            on_update=send_update
        )

        result_queue.put(
            ("worker_complete", reviewed_result)
        )

    except Exception:
        # Do not send internal exception details to the UI.
        result_queue.put(
            ("worker_error", None)
        )


# =========================================================
# REVIEWER STATUS POLLING
# =========================================================

@st.fragment(run_every=1)
def reviewer_status_panel():

    result_queue = st.session_state.get("review_queue")

    if result_queue is not None:

        # Drain all currently available queue messages.
        while True:

            try:
                stage, data = result_queue.get_nowait()

            except queue.Empty:
                break

            if stage == "review_status":

                st.session_state["review_status"] = "running"

            elif stage == "review_complete":

                st.session_state["review_status"] = (
                    "approved"
                    if data.get("approved")
                    else "completed"
                )

            elif stage == "worker_complete":

                st.session_state["reviewed_result"] = data

                st.session_state["review_status"] = (
                    "approved"
                    if data.get("approved")
                    else "completed"
                )

            elif stage == "worker_error":

                st.session_state["review_status"] = "error"

    status = st.session_state.get("review_status")

    if status == "running":

        st.info(
            "🔍 Reviewer is checking the study material in the background..."
        )

    elif status == "approved":

        st.success(
            "✅ Reviewer has completed the checks. Study material approved."
        )

    elif status == "completed":

        st.success(
            "✅ Reviewer has finished checking the study material."
        )

    elif status == "error":

        st.warning(
            "Reviewer could not complete the checks. "
            "Your generated Summary and Quiz are still available."
        )

    elif status == "not_started":

        st.caption("Reviewer has not started yet.")


# =========================================================
# INITIAL LANDING SCREEN
# =========================================================

if (
    st.session_state["result"] is None
    and "pending_document" not in st.session_state
):

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.markdown(
            '<p class="landing-caption">'
            'Upload your study material and let your AI agents '
            'create a personalized learning experience.'
            '</p>',
            unsafe_allow_html=True
        )

        st.write("")

        uploaded_file = st.file_uploader(
            "Upload your document",
            type=["pdf", "txt", "md"],
            help="Supported formats: PDF, TXT and Markdown.",
            key="initial_upload"
        )

        topic = st.text_input(
            "What would you like to understand?",
            placeholder="e.g. Machine Learning, Neural Networks, RAG...",
            key="initial_topic"
        )

        st.write("")

        generate = st.button(
            "✨ Generate My Study Pack",
            type="primary",
            use_container_width=True
        )

        if generate:

            if uploaded_file is None:

                st.warning("Please upload a document first.")

            else:

                st.session_state["pending_document"] = {
                    "bytes": uploaded_file.getvalue(),
                    "name": uploaded_file.name,
                    "topic": topic
                }

                st.rerun()

    st.stop()


# =========================================================
# SIDEBAR
# Appears after the initial submission.
# =========================================================

if (
    st.session_state["result"] is not None
    or "pending_document" in st.session_state
):

    with st.sidebar:

        st.header("📚 Study Assistant")

        st.caption(
            "Create a new study pack whenever you want."
        )

        st.divider()

        new_file = st.file_uploader(
            "Upload a new document",
            type=["pdf", "txt", "md"],
            key="new_upload"
        )

        new_topic = st.text_input(
            "Study topic",
            placeholder="Enter a topic (optional)",
            key="new_topic"
        )

        if st.button(
            "✨ Generate Study Pack",
            type="primary",
            use_container_width=True
        ):

            if new_file is None:

                st.warning("Please upload a document first.")

            else:

                st.session_state["pending_document"] = {
                    "bytes": new_file.getvalue(),
                    "name": new_file.name,
                    "topic": new_topic
                }

                # Clear previous study results.
                keys_to_clear = [
                    "result",
                    "assistant",
                    "feedback",
                    "review_queue",
                    "review_thread",
                    "reviewed_result",
                    "quiz_attempt",
                    "quiz_submitted",
                ]

                for key in keys_to_clear:
                    st.session_state.pop(key, None)

                st.session_state["review_status"] = "not_started"
                st.session_state["active_section"] = "Summary"

                # Clear previous quiz selections.
                for key in list(st.session_state.keys()):

                    if key.startswith("q") and key[1:].isdigit():
                        st.session_state.pop(key, None)

                st.rerun()

        st.divider()

        st.caption("Powered by Groq LLM")


# =========================================================
# STUDY WORKSPACE
# =========================================================

if st.session_state["result"] is not None:

    assistant = st.session_state["assistant"]
    result = st.session_state["result"]

    summary = result["summary"]
    quiz = result["quiz"]

    # -----------------------------------------------------
    # NAVIGATION BUTTONS
    # -----------------------------------------------------

    st.divider()

    nav_summary, nav_quiz = st.columns(2)

    with nav_summary:

        if st.button(
            "📝 Summary",
            type=(
                "primary"
                if st.session_state["active_section"] == "Summary"
                else "secondary"
            ),
            use_container_width=True
        ):

            st.session_state["active_section"] = "Summary"
            st.rerun()

    with nav_quiz:

        if st.button(
            "❓ Quiz",
            type=(
                "primary"
                if st.session_state["active_section"] == "Quiz"
                else "secondary"
            ),
            disabled=not bool(quiz),
            use_container_width=True
        ):

            st.session_state["active_section"] = "Quiz"
            st.rerun()

    st.divider()

    # -----------------------------------------------------
    # REVIEWER STATUS
    # -----------------------------------------------------

    reviewer_status_panel()

    st.divider()

    # -----------------------------------------------------
    # SUMMARY SECTION
    # -----------------------------------------------------

    if st.session_state["active_section"] == "Summary":

        render_summary(summary)

    # -----------------------------------------------------
    # QUIZ SECTION
    # -----------------------------------------------------

    elif st.session_state["active_section"] == "Quiz":

        st.subheader("Test Your Understanding")

        st.caption(
            f"Answer the following {len(quiz)} questions "
            "to evaluate your understanding."
        )

        # Keep an independent snapshot of the quiz being attempted.
        if st.session_state["quiz_attempt"] is None:

            st.session_state["quiz_attempt"] = [
                dict(question) for question in quiz
            ]

        attempted_quiz = st.session_state["quiz_attempt"]

        with st.form("quiz_form"):

            picks = []

            for i, question in enumerate(attempted_quiz):

                picks.append(
                    st.radio(
                        f"Q{i + 1}. {question['question']}",
                        question["options"],
                        index=None,
                        key=f"q{i}"
                    )
                )

            submitted = st.form_submit_button(
                "Submit Answers",
                type="primary"
            )

        if submitted:

            answers = [
                question["options"].index(pick)
                if pick is not None
                else None
                for question, pick in zip(attempted_quiz, picks)
            ]

            try:

                with st.spinner("Evaluating your answers..."):

                    feedback = assistant.evaluate(
                        answers,
                        quiz=attempted_quiz
                    )

                    st.session_state["feedback"] = feedback
                    st.session_state["quiz_submitted"] = True

            except Exception:

                st.error(
                    "Unable to evaluate your answers. Please try again."
                )

        # -------------------------------------------------
        # FEEDBACK
        # -------------------------------------------------

        feedback = st.session_state.get("feedback")

        if feedback:

            st.divider()

            st.subheader("📊 Your Performance")

            st.metric(
                "Your Score",
                f"{feedback['score']}/{feedback['total']}"
            )

            st.progress(
                max(
                    0.0,
                    min(1.0, feedback["percent"] / 100)
                )
            )

            st.caption(f"{feedback['percent']}%")

            st.markdown("### Overall Feedback")

            st.write(feedback["overall"])

            st.markdown("### Question-wise Feedback")

            for i, detail in enumerate(feedback["details"], 1):

                if detail["correct"]:

                    st.success(
                        f"Q{i}: Correct!\n\n"
                        f"{detail['feedback']}"
                    )

                else:

                    st.error(
                        f"Q{i}: Incorrect\n\n"
                        f"Your answer: {detail['your_answer']}\n\n"
                        f"Correct answer: {detail['correct_answer']}\n\n"
                        f"{detail['feedback']}"
                    )

            st.divider()

            st.markdown("### 📅 Your Personalized Study Plan")

            for step in feedback["study_plan"]:
                st.markdown(f"- {step}")

            with st.expander("View Answer Key"):

                for line in assistant.answer_key(attempted_quiz):
                    st.write(line)

    st.stop()


# =========================================================
# GENERATION SCREEN
# =========================================================

pending = st.session_state["pending_document"]

st.divider()

st.subheader("Your Study Material")

overall_status = st.empty()

try:

    document = load_document(
        pending["bytes"],
        pending["name"]
    )

    assistant = StudyAssistant()

    # -----------------------------------------------------
    # GENERATE SUMMARY AND QUIZ
    # -----------------------------------------------------

    with st.spinner(
        "AI agents are generating your Summary and Quiz..."
    ):

        result = assistant.generate_material(
            document,
            pending["topic"] or None
        )

    # -----------------------------------------------------
    # SAVE INITIAL MATERIAL
    # -----------------------------------------------------

    st.session_state["assistant"] = assistant
    st.session_state["result"] = result
    st.session_state["feedback"] = None
    st.session_state["document_name"] = pending["name"]

    st.session_state["quiz_attempt"] = None
    st.session_state["quiz_submitted"] = False
    st.session_state["active_section"] = "Summary"

    # -----------------------------------------------------
    # START REVIEWER IN BACKGROUND
    # -----------------------------------------------------

    result_queue = queue.Queue()

    review_thread = threading.Thread(
        target=reviewer_worker,
        args=(
            assistant,
            result["summary"],
            result["quiz"],
            result_queue,
        ),
        daemon=True,
    )

    st.session_state["review_queue"] = result_queue
    st.session_state["review_thread"] = review_thread
    st.session_state["review_status"] = "running"

    # Start the background thread.
    review_thread.start()

    # Remove pending document and display the workspace.
    st.session_state.pop("pending_document", None)

    st.rerun()


except DocumentError as e:

    st.error(str(e))
    st.session_state.pop("pending_document", None)

except LLMError as e:

    st.error(str(e))
    st.session_state.pop("pending_document", None)

except Exception:

    st.error(
        "Study pack generation failed. "
        "Please check your configuration and try again."
    )

    st.session_state.pop("pending_document", None)
