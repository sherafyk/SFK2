from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
import uuid
import json
from datetime import datetime
from .models import FieldDocument
from .vision import extract_document_data

app = FastAPI(title="Maritime Document Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted")

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    path = UPLOAD_DIR / f"{file_id}{ext}"
    content = await file.read()
    path.write_bytes(content)

    try:
        doc: FieldDocument = await run_in_threadpool(extract_document_data, path)
        result = {"id": file_id, "status": "completed", "data": doc.model_dump()}
        (UPLOAD_DIR / f"{file_id}_result.json").write_text(
            json.dumps(
                {
                    "id": file_id,
                    "filename": file.filename,
                    "processed_at": datetime.utcnow().isoformat(),
                    "data": doc.model_dump(),
                },
                indent=2,
            )
        )
        return result
    except Exception as e:
        return {"id": file_id, "status": "error", "error": str(e)}


@app.get("/api/result/{file_id}")
async def get_result(file_id: str):
    result_path = UPLOAD_DIR / f"{file_id}_result.json"
    if not result_path.exists():
        raise HTTPException(404, "Result not found")
    return json.loads(result_path.read_text())


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
