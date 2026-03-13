import hashlib
import os
import re
import tempfile

import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline import create_vectorstore, generate_content
from utils import parse_mcq_response, score_exam


st.set_page_config(page_title="StudySpark AI", layout="wide")
st.title("StudySpark AI | PDF Exam Generator")


def get_upload_signature(files):
    raw = "|".join(f"{f.name}:{f.size}" for f in files)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def extract_chunks(file_bytes: bytes, file_name: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        temp_path = tmp_file.name

    try:
        pages = PyPDFLoader(temp_path).load()
        chunks = splitter.split_documents(pages)
        return [
            {
                "page_content": c.page_content,
                "metadata": {**c.metadata, "source_file": file_name},
            }
            for c in chunks
        ]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def build_docs(files):
    raw_docs = []
    for uploaded in files:
        raw_docs.extend(extract_chunks(uploaded.getvalue(), uploaded.name))

    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in raw_docs
    ]


def clean_viva_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\*\*Question\s*(\d+)\*\*", r"Q\1:", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*\*Model Answer\*\*", "A:", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*\*Answer\*\*", "A:", cleaned, flags=re.IGNORECASE)
    return cleaned


uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    signature = get_upload_signature(uploaded_files)

    if st.session_state.get("upload_signature") != signature:
        with st.spinner("Reading PDFs ..."):
            docs = build_docs(uploaded_files)
            namespace = f"upload-{signature[:16]}"
            vectorstore = create_vectorstore(docs, namespace=namespace)

        st.session_state["upload_signature"] = signature
        st.session_state["vectorstore"] = vectorstore
        st.session_state["exam"] = None
        st.session_state["generated"] = {}
        st.success(f"Loaded {len(docs)} chunks from {len(uploaded_files)} PDFs")

    topic = st.text_input("Topic")
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    mcq_count = st.slider("Number of questions", 1, 20, 5)
    selected_outputs = st.multiselect(
        "Select outputs",
        ["MCQ", "Short Notes", "Flashcards", "Viva Q&A"],
        default=["MCQ"],
    )

    if st.button("Generate Content"):
        if not topic.strip():
            st.error("Please enter a topic.")
        elif not selected_outputs:
            st.error("Please select at least one output type.")
        else:
            st.session_state["generated"] = {}
            st.session_state["exam"] = None
            st.session_state["exam_topic"] = topic
            st.session_state["exam_difficulty"] = difficulty

            task_map = {
                "MCQ": "MCQ",
                "Short Notes": "Notes",
                "Flashcards": "Flashcards",
                "Viva Q&A": "Viva",
            }

            total_steps = len(selected_outputs)
            progress_text = st.empty()
            progress_bar = st.progress(0)

            for step, label in enumerate(selected_outputs, start=1):
                progress_text.info(f"Generating {label} ({step}/{total_steps})...")
                task = task_map[label]
                raw_output = generate_content(
                    task=task,
                    topic=topic,
                    vectorstore=st.session_state["vectorstore"],
                    difficulty=difficulty,
                    mcq_count=mcq_count,
                )

                if task == "MCQ":
                    try:
                        parsed_mcq = parse_mcq_response(raw_output)
                        st.session_state["generated"][task] = parsed_mcq
                        st.session_state["exam"] = parsed_mcq
                    except ValueError as exc:
                        st.error(f"Could not parse MCQ output: {exc}")
                        st.code(raw_output[:1500])
                else:
                    st.session_state["generated"][task] = raw_output

                progress_bar.progress(step / total_steps)

            progress_text.empty()
            st.success("Content generated.")


generated = st.session_state.get("generated", {})
if generated:
    if "Notes" in generated:
        st.subheader("Short Notes")
        st.markdown(generated["Notes"])
        st.download_button("Download Short Notes", generated["Notes"], "short_notes.txt")

    if "Flashcards" in generated:
        st.subheader("Flashcards")
        st.text(generated["Flashcards"])
        st.download_button("Download Flashcards", generated["Flashcards"], "flashcards.txt")

    if "Viva" in generated:
        st.subheader("Viva Q&A")
        viva_text = clean_viva_text(generated["Viva"])
        st.text(viva_text)
        st.download_button("Download Viva Q&A", viva_text, "viva_qa.txt")


exam = st.session_state.get("exam")
if exam:
    st.subheader("Exam")
    st.write(f"Topic: **{st.session_state.get('exam_topic', '-') }**")
    st.write(f"Difficulty: **{st.session_state.get('exam_difficulty', '-') }**")

    with st.form("exam_form"):
        selected_answers = {}
        for i, q in enumerate(exam):
            st.markdown(f"### Q{i + 1}. {q['question']}")
            selected_answers[i] = st.radio(
                f"Select answer for Q{i + 1}",
                q["choices"],
                index=None,
                key=f"answer_{i}",
            )

        submitted = st.form_submit_button("Submit Exam")

    if submitted:
        missing = [str(i + 1) for i, answer in selected_answers.items() if answer is None]
        if missing:
            st.error(f"Please answer all questions. Missing: {', '.join(missing)}")
        else:
            score, total, results = score_exam(exam, selected_answers)
            st.success(f"Score: {score}/{total}")

            lines = []
            for i, item in enumerate(results, start=1):
                st.markdown(f"**Q{i}. {item['question']}**")
                st.write(f"Your Answer: {item['user_answer']}")
                st.write(f"Correct Answer: {item['correct_answer']}")
                st.write(f"Explanation: {item['explanation']}")
                st.write("---")

                lines.append(
                    f"Q{i}: {item['question']}\n"
                    f"Your Answer: {item['user_answer']}\n"
                    f"Correct Answer: {item['correct_answer']}\n"
                    f"Explanation: {item['explanation']}"
                )

            st.download_button(
                "Download Results",
                data="\n\n".join(lines),
                file_name="exam_results.txt",
            )
