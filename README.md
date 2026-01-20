# Medical Chatbot Backend

A RAG (Retrieval Augmented Generation) powered medical chatbot API built with FastAPI, Pinecone, and Ollama.

## Features

- 🏥 Medical knowledge base powered by Pinecone vector database
- 🔄 Real-time streaming responses
- 🚀 FastAPI backend with CORS support
- 🎯 Context-aware responses using RAG

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```
PINECONE_API_KEY=your_pinecone_api_key_here
```

3. Run the server:
```bash
uvicorn rag_llm_stream:app --host 0.0.0.0 --port 8000
```

4. Open `chat.html` in your browser to test the chatbot.

## Deployment on Render

### Option 1: Using Blueprint (render.yaml)

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New" → "Blueprint"
4. Connect your GitHub repository
5. Add environment variable in Render:
   - `PINECONE_API_KEY`: Your Pinecone API key

### Option 2: Manual Deployment

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: medical-chatbot-api
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn rag_llm_stream:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `PINECONE_API_KEY`: Your Pinecone API key
6. Click "Create Web Service"

## Environment Variables

- `PINECONE_API_KEY`: Your Pinecone API key (required)

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /v1/chat/completions` - Chat completions with streaming support

## Tech Stack

- FastAPI - Web framework
- Pinecone - Vector database
- LangChain - RAG orchestration
- HuggingFace - Embeddings
- Ollama - LLM inference
