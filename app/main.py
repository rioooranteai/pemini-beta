from app.api.router import api_router
from app.core.middleware import register_middleware
from app.core.startup import lifespan
from fastapi import FastAPI


def create_app() -> FastAPI:

    app = FastAPI(
        title="RAG Chatbot Engine",
        version="1.0.0",
        lifespan=lifespan
    )

    register_middleware(app)

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        return {"status": "healthy", "message": "System is running fast and smoothly!"}

    return app


app = create_app()
