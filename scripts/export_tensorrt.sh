#!/bin/bash

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $(basename $0) model.onnx"
    exit 1
fi

ONNX_PATH=$1

if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX model not found"
    exit 1

fi

ENGINE_PATH="${ONNX_PATH%.*}.trt"

echo "Converting ONNX to TensorRT..."

/usr/src/tensorrt/bin/trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --explicitBatch \
  --fp16


echo "Done! Engine saved to $ENGINE_PATH!!"