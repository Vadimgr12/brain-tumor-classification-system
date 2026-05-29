from pydantic import BaseModel


class MriBrainResponse(BaseModel):
    class_name: int
    message: str
    probability: float
    end_to_end_latency_s: float
