import fire
from PIL import Image
from services.triton_client import infer
import numpy as np


def inference(image_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    triton_answer = infer(image)
    return triton_answer


if __name__ == "__main__":
    fire.Fire(inference)
