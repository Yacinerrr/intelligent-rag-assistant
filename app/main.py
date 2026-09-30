import os
import hmac
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from app.rag_pipeline import ask_question, new_conversation, list_conversations, get_conversation

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "")
SECRET_KEY = os.environ.get("SECRET_KEY", "")

if not ACCESS_PASSWORD:
    print("WARNING: ACCESS_PASSWORD not set in .env — login will always fail.")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set in .env — run the token generation command and add it.")

app = FastAPI(title="CERIST AI")

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def require_auth(request: Request):
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")


class LoginRequest(BaseModel):
    password: str


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    title: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/login")
def login(req: LoginRequest, request: Request):
    if not ACCESS_PASSWORD or not hmac.compare_digest(req.password, ACCESS_PASSWORD):
        raise HTTPException(status_code=401, detail="Incorrect password")
    request.session["authenticated"] = True
    return {"ok": True}


@app.post("/api/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.post("/api/conversations", dependencies=[Depends(require_auth)])
def create_conversation():
    conv = new_conversation()
    return {"id": conv["id"], "title": conv["title"]}


@app.get("/api/conversations", dependencies=[Depends(require_auth)])
def get_conversations():
    return list_conversations()


@app.get("/api/conversations/{conv_id}", dependencies=[Depends(require_auth)])
def get_conversation_detail(conv_id: str):
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_auth)])
def chat(req: ChatRequest):
    session_id = req.session_id or new_conversation()["id"]
    result = ask_question(req.question, session_id=session_id)
    return ChatResponse(**result)


@app.get("/login")
def login_page():
    return FileResponse(str(STATIC_DIR / "login.html"))


@app.get("/")
def index(request: Request):
    if not request.session.get("authenticated"):
        return RedirectResponse(url="/login")
    return FileResponse(str(STATIC_DIR / "index.html"))