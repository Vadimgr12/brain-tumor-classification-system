from fastapi import APIRouter, UploadFile, File
from services.triton_client import infer
from api.schemas import MriBrainResponse

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=MriBrainResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    triton_answer = infer(contents)

    return MriBrainResponse(
        class_name=triton_answer["class"],
        message=triton_answer["message"],
        probability=triton_answer["probability"],
        end_to_end_latency_s=triton_answer["end_to_end_latency_s"],
    )
