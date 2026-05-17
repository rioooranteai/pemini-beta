from fastapi import APIRouter
from app.api.routers.ingest import router as ingest_router
from app.api.routers.retrieval import router as retrieval_router
from app.api.routers.chat import router as chat_router

api_router = APIRouter()

api_router.include_router(ingest_router)
api_router.include_router(retrieval_router)
api_router.include_router(chat_router)