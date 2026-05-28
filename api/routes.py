from fastapi import APIRouter, UploadFile, File
from services.triton_client import infer

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    triton_answer = infer(contents)

    return triton_answer
