from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import json
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

OLLAMA_CHAT_URL = "https://ollamaapi.sharelive.site/v1/chat/completions"

# ------------------------
# APP
# ------------------------
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
# STREAM GENERATOR
# ------------------------
def ollama_stream(payload, user_query, retriever):
    # 1️⃣ Send immediate acknowledgment
    yield "data: {}\n\n"  # Keep connection alive
    
    # 2️⃣ Retrieve context (this takes time)
    print(f"🔍 Retrieving context for: {user_query}")
    docs = retriever.invoke(user_query)
    context = "\n\n".join(d.page_content for d in docs)
    print(f"✅ Context retrieved: {len(context)} chars from {len(docs)} documents")
    
    # 2a️⃣ Send sources metadata before streaming response
    sources = []
    for i, doc in enumerate(docs, 1):
        source_info = {
            "index": i,
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata
        }
        sources.append(source_info)
        print(f"   📄 Source {i}: {doc.metadata}")
    
    # Send sources as a special metadata chunk
    sources_chunk = {
        "id": "chatcmpl-rag-sources",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"sources": sources},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(sources_chunk)}\n\n"
    
    # 3️⃣ Inject system prompt into messages
    system_message = {
        "role": "system",
        "content": SYSTEM_TEMPLATE.format(context=context)
    }
    payload["messages"] = [system_message] + payload["messages"]
    
    # 4️⃣ Stream from Ollama
    print(f"🚀 Calling Ollama API...")
    with requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()

        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            # Ollama sends: "data: {...}"
            if raw_line.startswith("data:"):
                raw_line = raw_line.removeprefix("data:").strip()

            # Ignore keepalive / end markers
            if raw_line in ("[DONE]", ""):
                yield "data: [DONE]\n\n"
                break

            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                # Skip malformed / partial frames safely
                continue

            # Ollama end condition
            if chunk.get("done"):
                yield "data: [DONE]\n\n"
                break

            token = (
                chunk
                .get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
            )

            if not token:
                continue

            openai_chunk = {
                "id": "chatcmpl-rag-ollama",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }

            yield f"data: {json.dumps(openai_chunk)}\n\n"

# ------------------------
# CHAT COMPLETIONS (STREAMING)
# ------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body["messages"]
    user_query = messages[-1]["content"]
    
    print(f"\n📥 Received query: {user_query}")

    payload = {
        "model": body.get("model", "llama3.2:3b"),
        "messages": messages,  # Will be updated in generator
        "temperature": 0.2,
        "stream": True
    }

    return StreamingResponse(
        ollama_stream(payload, user_query, retriever),
        media_type="text/event-stream"
    )
