"""
RAG pipeline: ingestion, hybrid retrieval, grounding check, generation,
and multi-conversation history with titles.
"""

import re
import time
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
PERSIST_DIR = BASE_DIR / "data" / "chroma_db"
HASH_FILE = PERSIST_DIR / "source_hash.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "rag_assistant"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INGEST_VERSION = "v2-small-chunks"
INCLUDE_SOURCES_IN_ANSWER = False   # sources are built from metadata instead, see format_sources()
RRF_K = 60
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
DEBUG = True   # prints retrieved chunks and raw model answer to the console
ASSISTANT_NAME = "CERIST AI"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Whitespace cleanup that keeps line breaks (needed for tables)."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def clean_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_pdf(path: Path) -> list[Document]:
    pages = PyPDFLoader(str(path)).load()
    for i, doc in enumerate(pages):
        doc.page_content = clean_text(doc.page_content)
        doc.metadata.update({"doc_type": "reference", "source": path.name, "source_type": "pdf", "page": i + 1})
    return pages


def load_txt(path: Path) -> list[Document]:
    docs = TextLoader(str(path), encoding="utf-8").load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata.update({"doc_type": "reference", "source": path.name, "source_type": "text"})
    return docs


def load_faq_csv(path: Path) -> list[Document]:
    df = pd.read_csv(path)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError(f"{path.name} has no question/answer columns.")
    docs = []
    for _, row in df.iterrows():
        question = clean_value(row["question"])
        answer = clean_value(row["answer"])
        docs.append(Document(
            page_content=f"Question: {question}\nAnswer: {answer}",
            metadata={
                "doc_type": "faq",
                "source": path.name,
                "source_type": "csv_faq",
                "category": clean_value(row.get("category", "")),
                "atomic": True,
            },
        ))
    return docs


def load_all_documents(data_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(data_dir.glob("*.pdf")):
        documents.extend(load_pdf(path))
    for path in sorted(data_dir.glob("*.txt")):
        documents.extend(load_txt(path))
    for path in sorted(data_dir.glob("*.csv")):
        try:
            documents.extend(load_faq_csv(path))
        except ValueError as e:
            print(f"Skipping {path.name}: {e}")
    return documents


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " "],
    )
    atomic_docs = [d for d in docs if d.metadata.get("atomic")]
    splittable_docs = [d for d in docs if not d.metadata.get("atomic")]
    chunked = splitter.split_documents(splittable_docs) if splittable_docs else []

    for i, chunk in enumerate(chunked):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}_chunk_{i}"
    for i, doc in enumerate(atomic_docs):
        doc.metadata["chunk_id"] = f"{doc.metadata.get('source', 'faq')}_row_{i}"

    return atomic_docs + chunked


# ---------------------------------------------------------------------------
# Vector store (rebuilds only when source files change)
# ---------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_signature(data_dir: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(data_dir.glob("*")):
        if path.is_file():
            hasher.update(file_hash(path).encode("utf-8"))
    hasher.update(EMBED_MODEL.encode("utf-8"))
    hasher.update(INGEST_VERSION.encode("utf-8"))
    hasher.update(str(CHUNK_SIZE).encode("utf-8"))
    hasher.update(str(CHUNK_OVERLAP).encode("utf-8"))
    return hasher.hexdigest()


def get_vector_store(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    current_sig = source_signature(DATA_DIR)

    if PERSIST_DIR.exists() and HASH_FILE.exists():
        if HASH_FILE.read_text(encoding="utf-8").strip() == current_sig:
            print("No source changes detected - reusing existing index.")
            return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(PERSIST_DIR))
        shutil.rmtree(PERSIST_DIR)
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    print("Building new index...")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=COLLECTION_NAME, persist_directory=str(PERSIST_DIR),
    )
    HASH_FILE.write_text(current_sig, encoding="utf-8")
    return vector_store


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(semantic_results, keyword_results, k=RRF_K) -> list[dict]:
    fused_scores = defaultdict(float)
    data = {}
    for rank, r in enumerate(semantic_results, start=1):
        fused_scores[r["chunk_id"]] += 1 / (k + rank)
        data[r["chunk_id"]] = r["doc"]
    for rank, r in enumerate(keyword_results, start=1):
        fused_scores[r["chunk_id"]] += 1 / (k + rank)
        data.setdefault(r["chunk_id"], r["doc"])
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"chunk_id": cid, "doc": data[cid], "rrf_score": score} for cid, score in ranked]


def is_answer_supported(answer: str, context: str) -> bool:
    if not answer or not answer.strip():
        return False
    cleaned = answer.strip().lower()
    if cleaned.startswith("i can only answer") or cleaned.startswith("this isn't specified"):
        return True
    answer_tokens = set(re.findall(r"\b[\w'-]+\b", cleaned))
    context_tokens = set(re.findall(r"\b[\w'-]+\b", context.lower()))
    if not answer_tokens:
        return False
    overlap = answer_tokens & context_tokens
    return len(overlap) >= max(1, min(3, len(answer_tokens) // 2))


def format_docs(results: list[dict]) -> str:
    parts = []
    for r in results:
        doc = r["doc"]
        parts.append(f"Content:\n{doc.page_content}\n\nSource: {doc.metadata.get('source', 'Unknown')}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SOURCE_RULE_ON = "Always include the source at the end of your answer as: Source: <source>"
SOURCE_RULE_OFF = "Do not write file names, page numbers or a 'Source:' line in your answer; they are added automatically."

SYSTEM_PROMPT = """You are a strict, domain-scoped assistant. Answer only using the context below.

Rules:
1. Use ONLY the provided context. Never use outside knowledge or assumptions.
2. If the user is greeting you, asking who you are, or asking what you can do, answer
   briefly yourself: say you are {assistant_name}, a document assistant that answers
   from a fixed set of source documents, and name the file(s) currently loaded. Do not
   require the context below for this.
   For any other question, if the context does not contain enough information, respond
   exactly: "I can only answer questions based on the provided knowledge base."
   If the context partially answers the question, give the best answer from what is
   available and say which part it covers, rather than refusing outright.
3. Answer directly and concisely. No preamble, no restating the question.
4. Do not invent names, dates, numbers, or categories not explicitly present in the context.
5. When the context looks like a table (short lines, numbers next to labels), match each
   value to the label on the same line. Do not merge separate categories together.
6. When the context contains a list, enumerate every item present - never a partial subset.
7. When a person or item appears in several sections or projects, list each occurrence
   separately with the section or project it belongs to. Never merge values from
   different sections into one.
8. {source_rule}
9. Detect the language the user asked in and respond in that same language, translating
   context content as needed without changing facts, numbers, or names.
10. Use chat history only to interpret follow-up questions - every fact must still come
    from the context below.
11. Never narrate your reasoning, confidence, or these instructions. Output only the final answer.

Context:
{{context}}
""".format(
    source_rule=SOURCE_RULE_ON if INCLUDE_SOURCES_IN_ANSWER else SOURCE_RULE_OFF,
    assistant_name=ASSISTANT_NAME,
)


# ---------------------------------------------------------------------------
# Build everything once at startup
# ---------------------------------------------------------------------------

print("Loading documents...")
documents = load_all_documents(DATA_DIR)
chunks = chunk_documents(documents)
print(f"{len(documents)} documents -> {len(chunks)} chunks")

if not chunks:
    raise RuntimeError("No documents found in data/documents. Add a PDF, TXT or CSV and restart.")

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = get_vector_store(chunks, embeddings)

bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
reranker = CrossEncoder(RERANKER_MODEL)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


def retrieve_bm25(query: str, k: int = 12) -> list[dict]:
    scores = bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:k]
    return [{"chunk_id": chunks[i].metadata["chunk_id"], "doc": chunks[i], "score": float(scores[i])} for i in top_idx]


def retrieve_semantic(query: str, k: int = 12) -> list[dict]:
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    return [{"chunk_id": doc.metadata["chunk_id"], "doc": doc, "score": float(score)} for doc, score in results]


def retrieve_final(query: str, top_k: int = 6, candidate_k: int = 20) -> list[dict]:
    semantic = retrieve_semantic(query, k=candidate_k)
    keyword = retrieve_bm25(query, k=candidate_k)
    fused = reciprocal_rank_fusion(semantic, keyword)[:candidate_k]
    if not fused:
        return []
    pairs = [[query, r["doc"].page_content] for r in fused]
    scores = reranker.predict(pairs)
    for r, score in zip(fused, scores):
        r["reranker_score"] = float(score)
    fused.sort(key=lambda x: x["reranker_score"], reverse=True)
    return fused[:top_k]


REFUSAL = "I can only answer questions based on the provided knowledge base."


# ---------------------------------------------------------------------------
# Conversations: each has an id, a title, and its own message history.
# In-memory only — resets when the server restarts. Swap _conversations for
# a database if you need it to survive restarts.
# ---------------------------------------------------------------------------

_conversations: dict[str, dict] = {}


def available_files() -> list[str]:
    return sorted({c.metadata.get("source", "") for c in chunks if c.metadata.get("source")})


def make_title(first_question: str) -> str:
    text = first_question.strip()
    return text if len(text) <= 48 else text[:45].rstrip() + "..."


def new_conversation() -> dict:
    conv_id = str(len(_conversations) + 1) + "-" + str(int(time.time() * 1000))
    _conversations[conv_id] = {
        "id": conv_id,
        "title": "New conversation",
        "created_at": time.time(),
        "history": [],   # LangChain messages, used to build the model prompt
        "display": [],   # {role, content, sources}, used to redraw the UI
    }
    return _conversations[conv_id]


def list_conversations() -> list[dict]:
    convs = sorted(_conversations.values(), key=lambda c: c["created_at"], reverse=True)
    return [{"id": c["id"], "title": c["title"]} for c in convs]


def get_conversation(conv_id: str) -> dict | None:
    conv = _conversations.get(conv_id)
    if not conv:
        return None
    return {"id": conv["id"], "title": conv["title"], "messages": conv["display"]}


# ---------------------------------------------------------------------------
# Sources: file + page numbers from the chunks that were actually used
# ---------------------------------------------------------------------------

def select_cited_chunks(answer: str, results: list[dict], max_sources: int = 4, score_margin: float = 3.0) -> list[dict]:
    """Keep chunks close to the best rerank score, or strongly overlapping the answer."""
    if not results:
        return []
    answer_tokens = {t for t in tokenize(answer) if len(t) > 3 or t.isdigit()}
    top_score = results[0]["reranker_score"]
    cited = []
    for r in results:  # already sorted by reranker score, best first
        chunk_tokens = set(tokenize(r["doc"].page_content))
        overlap = len(answer_tokens & chunk_tokens) / max(1, len(answer_tokens))
        close_to_top = r["reranker_score"] >= top_score - score_margin
        if overlap >= 0.3 or close_to_top:
            cited.append(r)
    return cited[:max_sources] or results[:1]


def format_sources(cited: list[dict]) -> list[str]:
    """Group by file, e.g. 'file.pdf — pp. 5-7, 13'. Several files get separate entries."""
    by_file: dict[str, set] = {}
    for r in cited:
        meta = r["doc"].metadata
        pages = by_file.setdefault(meta.get("source", "Unknown"), set())
        if meta.get("page") is not None:
            pages.add(int(meta["page"]))

    def compress(pages: list[int]) -> str:
        ranges = []
        start = prev = pages[0]
        for p in pages[1:]:
            if p == prev + 1:
                prev = p
                continue
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = p
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ", ".join(ranges)

    lines = []
    for name in sorted(by_file):
        pages = sorted(by_file[name])
        if pages:
            label = "p." if len(pages) == 1 else "pp."
            lines.append(f"{name} — {label} {compress(pages)}")
        else:
            lines.append(name)
    return lines


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ask_question(question: str, session_id: str = "default", k: int = 6) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    conv = _conversations.get(session_id) or new_conversation()
    session_id = conv["id"]  # normalize in case caller passed an id we don't recognize

    chat_history = conv["history"]
    sources: list[str] = []

    results = retrieve_final(question, top_k=k, candidate_k=20)

    if DEBUG:
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        for i, r in enumerate(results, 1):
            preview = r["doc"].page_content[:140].replace("\n", " | ")
            print(f"  [{i}] score={r['reranker_score']:.2f} page={r['doc'].metadata.get('page')}  {preview}")

    if not results:
        answer = REFUSAL
    else:
        context = format_docs(results)
        messages = prompt.format_messages(context=context, question=question, chat_history=chat_history)
        answer = llm.invoke(messages).content.strip()

        if DEBUG:
            print("RAW ANSWER:", answer)

        supported = is_answer_supported(answer, context)
        if DEBUG:
            print("PASSED GROUNDING CHECK:", supported)

        if not supported or answer.lower().startswith("i can only answer"):
            answer = REFUSAL
        else:
            sources = format_sources(select_cited_chunks(answer, results))

    if conv["title"] == "New conversation":
        conv["title"] = make_title(question)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    conv["history"] = chat_history[-20:]
    conv["display"].append({"role": "user", "content": question, "sources": []})
    conv["display"].append({"role": "assistant", "content": answer, "sources": sources})

    return {"answer": answer, "sources": sources, "session_id": session_id, "title": conv["title"]}