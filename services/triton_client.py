from services.preprocessing import preprocess
from services.postprocessing import postprocess
import tritonclient.http as httpclient
from fastapi import HTTPException

client = httpclient.InferenceServerClient(url="localhost:8000")


def infer(x):
    try:
        x = preprocess(x)

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"BAD IMAGE. Preprocessing error: {str(e)}"
        )

    try:
        infer_input = httpclient.InferInput("x", x.shape, "FP32")
        infer_input.set_data_from_numpy(x)
        requested_output = httpclient.InferRequestedOutput("linear")

        result = client.infer(
            model_name="brain_tumor_classification",
            inputs=[infer_input],
            outputs=[requested_output],
        )
        logits = result.as_numpy("linear")
        if logits is None:
            raise RuntimeError("Triton returned empty output")

    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"TRITON MODEL ERROR. Preprocessing error: {str(e)}"
        )
    try:
        answer = postprocess(logits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"POSTPROCESS ERROR: {str(e)}")

    return answer


if __name__ == "__main__":
    infer(input)
