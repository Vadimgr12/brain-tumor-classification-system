from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents),
    }
