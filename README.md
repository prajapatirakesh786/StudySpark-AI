# Exam Content Generator from PDFs

This project generates exam content from uploaded PDF files using a simple RAG pipeline.

## Features
- Upload one or more PDFs
- Generate content from a selected topic:
  - MCQ
  - Short Notes
  - Flashcards
  - Viva Q&A
- Choose difficulty level and number of questions
- Attempt MCQ quiz directly in the app
- Get score and download generated outputs

## Tech Stack
- Streamlit
- LangChain
- Pinecone
- Groq API
- Hugging Face embeddings

## Project Files
- `app.py` - Streamlit UI
- `rag_pipeline.py` - RAG + LLM logic
- `prompt_template.py` - prompts for MCQ, notes, flashcards, viva
- `utils.py` - JSON parsing and scoring
- `requirements.txt` - dependencies

## Run Locally
1. Create and activate virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` from `.env.example` and add keys
4. Run app:
   ```bash
   streamlit run app.py
   ```

## Required Environment Variables
- `PINECONE_API_KEY`
- `PINECONE_INDEX` (default: `mcqcreator`)
- `PINECONE_CLOUD` (default: `aws`)
- `PINECONE_REGION` (default: `us-east-1`)
- `GROQ_API_KEY`
- `GROQ_MODEL` (default: `llama-3.1-8b-instant`)
- `GROQ_BASE_URL` (default: `https://api.groq.com/openai/v1`)
- `HF_TOKEN` (optional)

## Notes
- Do not upload `.env` to GitHub.
- `.gitignore` is already configured.

## License
MIT
