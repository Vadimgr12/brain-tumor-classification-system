from services.preprocessing import preprocess
from services.postprocessing import postprocess
import tritonclient.http as httpclient
from fastapi import HTTPException
from services.constants import INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME

client = httpclient.InferenceServerClient(url="localhost:8000")


def infer(x):
    try:
        x = preprocess(x)

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"BAD IMAGE. Preprocessing error: {str(e)}"
        )

    try:
        infer_input = httpclient.InferInput(INPUT_TENSOR_NAME, x.shape, "FP32")
        infer_input.set_data_from_numpy(x)
        requested_output = httpclient.InferRequestedOutput(OUTPUT_TENSOR_NAME)

        result = client.infer(
            model_name="brain_tumor_classification",
            inputs=[infer_input],
            outputs=[requested_output],
        )
        logits = result.as_numpy(OUTPUT_TENSOR_NAME)
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
