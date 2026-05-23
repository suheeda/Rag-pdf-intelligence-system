import logging
from typing import List, Dict
from app.config import settings

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are a helpful assistant answering questions from a private document corpus.

Use ONLY the context below to answer. If the context does not contain the answer, say:
"I couldn't find relevant information in the provided documents."

Always end your answer with a "Sources:" section listing the document and page numbers used.

Context:
{context}

Question: {question}

Answer:"""


def build_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk['source']}, Page {chunk['page']}\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(question: str, chunks: List[Dict]) -> str:
    context = build_context(chunks)
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    provider = settings.LLM_PROVIDER.lower()

    if provider == "groq":
        return _call_groq(prompt)
    elif provider == "gemini":
        return _call_gemini(prompt)
    elif provider == "ollama":
        return _call_ollama(prompt)
    elif provider == "openai":
        return _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _call_groq(prompt: str) -> str:
    """
    Groq — free tier, no credit card, extremely fast inference.
    Get key at: https://console.groq.com  (takes 60 seconds)
    Recommended models (all free):
      - llama3-8b-8192    → fastest, good quality
      - llama3-70b-8192   → better quality, slightly slower
      - mixtral-8x7b-32768 → large context window
    """
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq error: {e}")
        raise


def _call_gemini(prompt: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 1024},
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise


def _call_ollama(prompt: str) -> str:
    """Fully local/offline — run: ollama pull mistral"""
    try:
        import httpx
        resp = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        raise


def _call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise
