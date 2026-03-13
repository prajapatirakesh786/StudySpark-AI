import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from prompt_template import get_prompt


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "mcqcreator")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
HF_TOKEN = os.getenv("HF_TOKEN")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")


@lru_cache
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": HF_TOKEN},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing in environment.")

    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0.2,
    )


@lru_cache
def get_pinecone_client():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing in environment.")
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index():
    pc = get_pinecone_client()
    indexes = pc.list_indexes()

    if hasattr(indexes, "names"):
        names = set(indexes.names())
    else:
        names = {item["name"] for item in indexes}

    if PINECONE_INDEX not in names:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


def create_vectorstore(docs, namespace):
    ensure_index()

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=get_embeddings(),
        namespace=namespace,
    )

    if docs:
        vectorstore.add_documents(docs)

    return vectorstore


def _get_context(topic, vectorstore, mcq_count):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": min(max(6, mcq_count * 2), 20), "fetch_k": 30},
    )
    docs = retriever.invoke(topic)
    return "\n\n".join(doc.page_content[:900] for doc in docs)


def generate_content(task, topic, vectorstore, difficulty, mcq_count=5):
    context = _get_context(topic, vectorstore, mcq_count)
    prompt_template = get_prompt(task)

    kwargs = {"context": context, "topic": topic, "difficulty": difficulty}
    if task == "MCQ":
        kwargs["mcq_count"] = mcq_count

    prompt = prompt_template.format(**kwargs)

    return get_llm().invoke(prompt).content


def generate_mcq(topic, vectorstore, mcq_count, difficulty):
    return generate_content("MCQ", topic, vectorstore, difficulty, mcq_count)
