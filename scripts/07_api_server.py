"""
Phase 5 - Task 5.1: FastAPI Inference Server.
Provides REST API endpoints for the fine-tuned CoT model via Ollama backend.

Usage:
  python scripts/07_api_server.py
  # or
  uvicorn scripts.07_api_server:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import time
import httpx
from typing import Optional
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "qwen-cot-0.8b")


# --- Request/Response Models ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    model: str = Field(default=DEFAULT_MODEL, description="Ollama model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    system: Optional[str] = Field(default=None, description="System prompt override")


class GenerateResponse(BaseModel):
    model: str
    response: str
    total_duration_ms: float
    eval_count: int
    eval_rate: float  # tokens/sec


class HealthResponse(BaseModel):
    status: str
    ollama: str
    model: str
    timestamp: float


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify Ollama connection
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            if DEFAULT_MODEL not in [m.split(":")[0] for m in models]:
                print(f"WARNING: Model '{DEFAULT_MODEL}' not found in Ollama. Available: {models}")
            else:
                print(f"Ollama connected. Model '{DEFAULT_MODEL}' available.")
    except Exception as e:
        print(f"WARNING: Cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}")
    yield


# --- App ---
app = FastAPI(
    title="LLM Optimization Pipeline - Inference API",
    description="REST API for the CoT-distilled Qwen3.5-0.8B model served via Ollama",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and Ollama backend health."""
    ollama_status = "disconnected"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                ollama_status = "connected"
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        ollama=ollama_status,
        model=DEFAULT_MODEL,
        timestamp=time.time(),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using the fine-tuned model via Ollama."""
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        },
    }
    if request.system:
        payload["system"] = request.system

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama backend is not running.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Generation timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total_ns = data.get("total_duration", 0)
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1)

    return GenerateResponse(
        model=data.get("model", request.model),
        response=data.get("response", ""),
        total_duration_ms=total_ns / 1e6,
        eval_count=eval_count,
        eval_rate=eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0,
    )


@app.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
