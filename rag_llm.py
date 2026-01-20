from fastapi import FastAPI, Request
import requests
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------
# ENV
# ------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"

# IMPORTANT: OpenAI-compatible endpoint
OLLAMA_CHAT_URL = "https://ollamaapi.sharelive.site/v1/chat/completions"

# ------------------------
# APP
# ------------------------
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------
# RAG INIT (ONCE)
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

SYSTEM_TEMPLATE = (
    "You are a medical assistant. Use the context below to answer the question. "
    "If the answer is not present, say you don't know.\n\n"
    "Context:\n{context}"
)

# ------------------------
# CHAT COMPLETIONS (RAG)
# ------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages = body["messages"]
    user_query = messages[-1]["content"]

    # 1️⃣ Retrieve context
    docs = retriever.invoke(user_query)
    context = "\n\n".join(d.page_content for d in docs)

    # 2️⃣ Inject system prompt
    system_message = {
        "role": "system",
        "content": SYSTEM_TEMPLATE.format(context=context)
    }

    final_messages = [system_message] + messages

    # 3️⃣ Forward to Ollama (OpenAI protocol)
    payload = {
        "model": body.get("model", "llama3.2:3b"),
        "messages": final_messages,
        "temperature": 0.2,
        "stream": False
    }

    r = requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=180  # CPU + 8B model needs this
    )
    r.raise_for_status()

    ollama_resp = r.json()
    content = ollama_resp["choices"][0]["message"]["content"]

    # 4️⃣ Return OpenAI-compatible response
    return {
        "id": "chatcmpl-rag-ollama",
        "object": "chat.completion",
        "created": 0,
        "model": payload["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ]
    }
