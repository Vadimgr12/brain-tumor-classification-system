# MRI Brain Tumor Classification System

An end-to-end MLOps pipeline for brain tumor classification from MRI images using a fine-tuned EfficientNetV2-S architecture pretrained on ImageNet. This system covers the full machine learning lifecycle: automated data ingestion, preprocessing, reproducible pipeline orchestration, experiment tracking, model export, and inference serving using multiple backends.
## Problem Statement
The objective of this project is to accurately classify brain MRI images into four distinct categories:
* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

To ensure stable and computationally feasible training, all experiments were performed on Apple Silicon (M1) using the MPS backend under constrained computational resources, without access to dedicated GPU clusters. Under these constraints, the task was formulated as an image classification problem instead of a more computationally expensive object detection setting. This choice significantly reduces training complexity while preserving meaningful diagnostic capability and enables a complete and reproducible MLOps pipeline covering training, evaluation, export, and deployment.

---

Markdown
## Dataset Specification
* **Source:** Kaggle (MRI Brain Tumor Dataset with Bounding Boxes)
* **Link:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)
* **Creator / Author:** Ahmed Sorour (Published: 2024)
* **Volume:** ~5,000 RGB MRI images + corresponding YOLO-formatted `.txt` annotation files (~140 MB total payload).
* **Processing Input:** Images are dynamically resized to $256 \times 256$ pixels and normalized according to calculated dataset-specific statistics using the `Albumentations` library.
* **Data Augmentations:** To enhance model generalization, prevent overfitting, and ensure robustness against clinical imaging variations, random spatial and color augmentations (such as flips, rotations, and brightness/contrast adjustments) are applied via `Albumentations` during the training phase.

###  Dataset Advantages & Key Strengths
* **Class Balance:** Unlike many medical datasets suffering from severe class imbalance, this dataset features a **highly balanced distribution** across all four target categories (glioma, meningioma, pituitary, and no tumor). This inherent balance eliminates the need for complex loss-weighting techniques or oversampling, enabling stable and unbiased objective function convergence.

### ⚠ Engineering Challenges & Solutions
* **Annotation Parsing:** The source dataset is structured for object detection, with label files containing bounding box coordinates (`[class_id, x_center, y_center, width, height]`). A custom preprocessing script was designed to robustly parse these `.txt` files, isolate the standalone `class_id` for each image, and seamlessly map them into a categorical classification format.
* **Rigorous Evaluation Strategy (Train/Val/Test Split):** To ensure a completely unbiased final performance evaluation, special care was taken to explicitly and cleanly isolate a **hidden Test set** alongside the standard Training and Validation sets (70/15/15 ratio). The splitting process enforces strict **stratification** to preserve identical class distributions across all three splits and uses a fixed random seed to guarantee complete pipeline reproducibility.
---

## Model Architecture
* **Backbone:** `EfficientNetV2-S`  pretrained on ImageNet, selected for its strong balance between accuracy and computational efficiency. EfficientNetV2-S is a lightweight variant of the EfficientNet family, designed to achieve high performance with significantly fewer parameters compared to larger architectures. This makes it particularly suitable for medical imaging tasks, where datasets are often limited in size and computational resources are constrained.
* The S (small) variant was chosen due to hardware limitations and the absence of large-scale GPU infrastructure. Despite its compact size, it retains strong representational power, which is critical for extracting subtle features in MRI brain scans. EfficientNet-based architectures are widely used in medical image analysis due to their strong transfer learning capabilities and robust performance on small to medium-sized datasets.
* **Fine-Tuning Strategy:** he model was fine-tuned using EfficientNetV2-S pretrained on ImageNet. During training, all feature extractor layers were initially frozen, and only the final **3 feature** blocks were unfrozen (n_unfrozen = 3), allowing task-specific adaptation while preserving pretrained representation.
* **Input Tensor:** `[Batch_Size, 3, 256, 256]` (RGB Image)
* **Output Layer:** A vector of raw logits mapped to the 4 target classes, converted to explicit prediction probabilities via a `Softmax` layer during inference.

---

## Performance Metrics & Validation

### Validation Strategy
* **Data Split:** Structured into **70% Training / 15% Validation / 15% Testing**.
* **Reproducibility:** Enforced via a fixed global random seed across all data loaders and operations.
* **Stratification:** Strict class stratification is applied during data splits to preserve identical target distributions across train, validation, and test subsets.

### Final Evaluation Results
The model demonstrates excellent generalization and high robustness, notably achieving perfect binary recall for tumor presence:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | `0.9849` |
| **Recall (Macro)** | `0.9827` |
| **F1-Score (Macro)** | `0.9844` |
| **Precision-Recall AUC (PR-AUC)** | `0.9944` |
| **ROC-AUC** | `0.9976` |
| **Binary Recall (Tumor vs. No Tumor)** | `1.0000` |

---

## Core System & MLOps Architecture

The system is fully decoupled into automated operational stages, enabling seamless execution from raw data to production serving:

```
[Raw MRI Data] ---> [DVC Preprocessing & Split] ---> [PyTorch Lightning Training]
                                                               |
   +-----------------------------------------------------------+
   |                                                           |
   v                                                           v
[MLflow Run Logs]                                      [Checkpoint Save] ------------- >[ONNX Serialization]
(Params, Metrics, Artifacts)                                                                |
                                                                                            v
                                                                             +--------------+--------------+
                                                                             v                             v
                                                                     [FastAPI Web Service]  <----+--->  [Triton Inference Server]
```

### 1. Training & Orchestration Pipeline (`DVC`)
The end-to-end pipeline is structured and version-controlled via **Data Version Control (DVC)**. Running `dvc repro` deterministically executes the following sequential stages:
1.  **Data Download & Extraction:** Pulls and verifies the raw source images.
2.  **Dataset Splitting:** Performs stratified splitting into train/val/test sets.
2.  **Preprocessing & Calculation:** Computes dataset-wide (on train dataset) normalization statistics and applies augmentations.
4.  **Model Training:** Executes training via **PyTorch Lightning** with automatic checkpointing based on validation loss.
5.  **Evaluation:** Evaluates the best model checkpoint against the test set, exporting metrics.
6.  **Model Export:** Converts the fine-tuned PyTorch weights into optimized ONNX format.
7.  **Triton Model Preparation:** Structures the generated ONNX file and write configuration rules into the local Model Repository format.

### 2. Experiment Tracking (`MLflow`)
Every pipeline run is tracked by **MLflow**. The system automatically logs:
* **Hyperparameters:** Learning rate.
* **Metrics:** Continuous training/validation loss curves, step-by-step accuracy, and final evaluation arrays.
* **Artifacts:** Best performing model checkpoints, confusion matrices.
![MLflow Training Tracking](images/mlflow1.png) *Figure 1: MLflow dashboard.*
### 3. Model Export (`ONNX`)
To decouple the model from the PyTorch runtime and accelerate inference, the trained checkpoint is serialized to **ONNX (Opset Version 17)**:
* **Input Node Name:** `x`
* **Output Node Name:** `logits`

---

## 🚀 Inference Ecosystem & Deployment

The architecture implements a centralized inference strategy where **NVIDIA Triton Inference Server acts as the core execution engine**. Both the Web API and the CLI do not run the model locally; instead, they serve as specialized client interfaces that preprocess data and communicate directly with the Triton server.

The system provides three production-ready execution paths and components:

### A. Scalable Production Serving Core (Triton Inference Server)
The central backbone of the inference ecosystem. It hosts the optimized ONNX model repository and handles all heavy tensor computations.
* **Concurrent Execution:** Enables multiple model instances to run simultaneously across incoming request streams.
* **Dynamic Batching:** Automatically groups individual incoming requests into optimal batch sizes on the fly, maximizing hardware throughput.
* **Standardized Endpoints:** Exposes high-performance, low-latency **gRPC** and **HTTP/REST** protocols.

### B. Lightweight REST API Gateway (FastAPI)
A highly optimized Python web service tailored for low-overhead client communication and public routing.
* **Proxy Architecture:** Acts as an API Gateway/BFF. It accepts raw binary image uploads from users, applies localized preprocessing (`Albumentations`), and **forwards the resulting tensors directly to Triton** via HTTP.
* **Response Payload:** Receives raw logits back from Triton, converts them into class probabilities via Softmax, and returns a structured JSON containing the predicted class, probabilities, and detailed end-to-end inference latency metrics.

### C. Command Line Interface (CLI)
A dedicated terminal utility designed for rapid diagnostics, automated batch scripts, or local engineering workflows.
* **Direct Triton Client:** It reads a local image path from terminal arguments, performs the required image transformations, **sends a synchronous inference request to the Triton server**, and prints the structured classification output directly to the console.
![FastAPI & Triton Inference Response](images/fastapi.png) *Figure 2: FastAPI interactive documentation demonstrating a successful brain tumor classification request.*
---

## ⚡ Quick Start & Pipeline Execution

> **Before running any training experiments**, ensure that the MLflow tracking server is active. All training runs, hyperparameters, metrics, and production artifacts are logged there for absolute reproducibility and downstream evaluation.

All project commands—spanning data processing, model training, evaluation, export, serving, and infrastructure management—are centralized inside the `Makefile`. **It is highly recommended to strictly use `make` targets** instead of executing underlying scripts manually to guarantee environment consistency and pipeline reproducibility.

### Example:
This will start the full MLOps infrastructure, including the MLflow tracking server, the complete DVC pipeline execution, and the launch of Triton Inference Server and MinIO services.
```bash
make run-mlflow
make dvc-repro
make run-gateway
make run-docker-all
```

## Technology Stack
* **Deep Learning Framework:** `PyTorch`, `PyTorch Lightning`
* **Computer Vision & Augmentation:** `Albumentations`
* **Pipeline & Data Versioning:** `DVC` (Data Version Control)
* **Experiment Metadata Server:** `MLflow`
* **Production Model Serving:** `Triton Inference Server`, `FastAPI`

---

## Key Architecture Decisions
1.  **Classification Shift:** Transitioning the problem statement from object detection to image classification drastically reduced computational overhead, enabling stable and efficient training on local environments (Apple Silicon / CPU) while preserving clinically meaningful diagnostic performance (tumor presence and type). This design choice was made to prioritize the development of a complete and reproducible MLOps pipeline, including training, experiment tracking, model export, and deployment, rather than focusing on computationally intensive detection modeling.
2.  **EfficientNet Transfer Learning:** Utilizing a highly optimized pretrained CNN allowed the system to hit over **98% accuracy** rapidly with a relatively small specialized dataset (~6,000 samples).
3.  **ONNX Portability:** Exporting checkpoints to ONNX ensures that the downstream deployment pipelines are completely independent of the training source code, enabling rapid execution across varying execution engines.

---
**Author:** Vadim Griaznov
