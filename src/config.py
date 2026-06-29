"""
config.py
─────────
Local LangChain configuration for the DevOps pipeline.

The workflow uses LangGraph state graphs with LangChain prompts and
`ChatOllama` as the shared local model.

Prerequisites
-------------
    pip install langchain langchain-ollama langgraph
    ollama pull llama3.2:latest  # or llama3.2:latest / llama3.1:70b / llama3:8b / etc.
    ollama serve                  # if not already running as a service

Environment variables (all optional, sensible defaults provided):
    OLLAMA_MODEL        - model tag to use            (default: "llama3.2:latest")
    OLLAMA_BASE_URL      - where the Ollama server is  (default: "http://localhost:11434")
    OLLAMA_TEMPERATURE   - sampling temperature         (default: 0.2)

Note on determinism for research:
    For prompt-injection-propagation experiments you generally want LOW
    temperature so that "did the injection change the output" isn't
    confounded by ordinary sampling noise. Default is kept low (0.2) for
    that reason — override with OLLAMA_TEMPERATURE if you need otherwise.
"""

import os
from functools import lru_cache
 

def _settings(model, base_url, temperature):
    model = model or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature = (
        temperature if temperature is not None else float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    )
    return model, base_url, temperature


@lru_cache(maxsize=None)
def get_langchain_llm(model: str = None, base_url: str = None, temperature: float = None):
    """Build (and cache) a LangChain `ChatOllama` instance."""
    from langchain_ollama import ChatOllama

    model, base_url, temperature = _settings(model, base_url, temperature)
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def get_llm(model: str = None, base_url: str = None, temperature: float = None):
    """Backward-compatible alias for `get_langchain_llm()`."""
    return get_langchain_llm(model=model, base_url=base_url, temperature=temperature)

