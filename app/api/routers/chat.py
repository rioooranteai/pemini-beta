from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.api.schemas import ChatRequest, ChatResponse, RetrievedChunkResponse
from app.engines.rag_engine import RAGEngine
from app.engines.rag_engine import ChatRequest as EngineChatRequest
from app.core.dependencies import get_rag_engine
import json

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, engine: RAGEngine = Depends(get_rag_engine)):
    result = await engine.chat(
        EngineChatRequest(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            system_prompt=request.system_prompt
        )
    )
    return ChatResponse(
        answer=result.answer,
        sources=[RetrievedChunkResponse(**s.__dict__) for s in result.sources]
    )


@router.post("/stream")
async def stream_chat(request: ChatRequest, engine: RAGEngine = Depends(get_rag_engine)):

    async def event_generator():
        sources = []
        async for token, src in engine.stream_chat(
            EngineChatRequest(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters,
                system_prompt=request.system_prompt
            )
        ):
            if token is not None:
                yield f"data: {json.dumps({'token': token})}\n\n"
            if src is not None:
                sources = [RetrievedChunkResponse(**s.__dict__).model_dump() for s in src]

        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")