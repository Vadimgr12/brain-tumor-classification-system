import numpy as np


def get_message(class_id, probability):
    MESSAGES = {
        0: "GLIOMA detected",
        1: "MENINGIOMA detected",
        2: "NO TUMOR detected",
        3: "PITUITARY detected",
    }

    base = MESSAGES[class_id]
    status = "Looks healthy." if class_id == 2 else "Visit a doctor ASAP."
    return f"{base} (Model probability = {probability:.4f}). {status}"


def softmax(logits):
    exp_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def postprocess(logits):
    if not isinstance(logits, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(logits)}")

    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {logits.shape}")

    if logits.shape[1] != 4:
        raise ValueError(f"Expected 4 classes in dim=1, got {logits.shape[1]}")

    probabilities = softmax(logits)
    class_id = np.argmax(probabilities, axis=1)[0]

    probability = probabilities[0, class_id]

    message = get_message(class_id, probability)

    return {
        "class": int(class_id),
        "message": message,
        "probability": float(probability),
    }
