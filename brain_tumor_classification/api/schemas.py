from pydantic import BaseModel


class MriBrainResponse(BaseModel):
    class_name: str
    confidence: float
