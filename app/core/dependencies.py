from app.engines.rag_engine import RAGEngine
from fastapi import Request


def get_rag_engine(request: Request) -> RAGEngine:
    return request.app.state.rag_engine
