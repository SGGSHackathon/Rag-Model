from fastapi import FastAPI, Request
import requests

app = FastAPI()   # ← MUST be here, top-level

OLLAMA_CHAT_URL = "https://ollamaapi.sharelive.site/api/chat"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    payload = {
        "model": body["model"],
        "messages": body["messages"],
        "stream": False
    }

    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    r.raise_for_status()

    ollama_resp = r.json()
    content = ollama_resp["message"]["content"]

    return {
        "id": "chatcmpl-ollama",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
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
