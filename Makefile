.PHONY: run-minio run-gateway run-triton run-docker-all down run-mlflow \
run-train run-test generate-triton-config dvc-init dvc-repro

# ------------------------
# Services
# ------------------------

run-minio:
	docker compose up -d minio

run-gateway:
	uv run uvicorn api.main:app --reload --port 8081

run-triton:
	docker compose up -d triton

run-docker-all:
	docker compose up -d

down:
	docker compose down -v

run-mlflow:
	uv run mlflow ui --port 8080

# ------------------------
# ML pipeline
# ------------------------

run-train:
	uv run python brain_tumor_classification/train.py

run-test:
	uv run python brain_tumor_classification/evaluate.py

generate-triton-config:
	uv run python scripts/make_triton_config.py

# ------------------------
# DVC pipeline
# ------------------------

dvc-init:
	dvc init
	uv run dvc remote add -d myremote s3://dvc-storage/ || true
	uv run dvc remote modify myremote endpointurl http://localhost:9000
	uv run dvc remote modify myremote access_key_id miniovadim
	uv run dvc remote modify myremote secret_access_key miniovadim

# Run full ML pipeline (train → evaluate → export → deploy artifacts) (if any dependency changed)
dvc-repro:
	uv run dvc repro
