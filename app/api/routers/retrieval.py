from app.api.schemas import RetrieveRequest, RetrieveResponse, RetrievedChunkResponse
from app.core.dependencies import get_rag_engine
from app.engines.rag_engine import RAGEngine
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])


@router.post("/", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest, engine: RAGEngine = Depends(get_rag_engine)):
    chunks = await engine.retrieve(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters
    )
    return RetrieveResponse(
        query=request.query,
        results=[RetrievedChunkResponse(**c.__dict__) for c in chunks]
    )
