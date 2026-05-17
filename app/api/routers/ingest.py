import json
from app.api.schemas import (
    IngestDocumentRequest, IngestBatchRequest, IngestResponse
)
from app.core.dependencies import get_rag_engine
from app.engines.rag_engine import RAGEngine
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

@router.post("/upload", summary="Unggah File Excel/PDF")
async def upload_document(file: UploadFile = File(...),
        metadata_json: str = Form(default="{}"),
        engine: RAGEngine = Depends(get_rag_engine)
):
    try:
        metadata_dict = json.loads(metadata_json)

        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Hanya file Excel yang diizinkan untuk MVP ini.")

        result = await engine.process_uploaded_file(
            filename=file.filename,
            file_stream=file.file,
            metadata=metadata_dict
        )

        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Format metadata_json tidak valid.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=IngestResponse)
async def ingest_document(request: IngestDocumentRequest, engine: RAGEngine = Depends(get_rag_engine)):
    result = await engine.ingest(
        doc_id=request.doc_id,
        content=request.content,
        metadata=request.metadata
    )
    if not result.success:
        raise HTTPException(status_code=500, detail=f"Failed to ingest doc: {request.doc_id}")
    return IngestResponse(**result.__dict__)


@router.post("/batch", response_model=list[IngestResponse])
async def ingest_batch(request: IngestBatchRequest, engine: RAGEngine = Depends(get_rag_engine)):
    results = await engine.ingest_batch([doc.__dict__ for doc in request.documents])
    return [IngestResponse(**r.__dict__) for r in results]


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, engine: RAGEngine = Depends(get_rag_engine)):
    await engine.delete_document(doc_id)
    return {"message": f"Document '{doc_id}' deleted successfully"}
