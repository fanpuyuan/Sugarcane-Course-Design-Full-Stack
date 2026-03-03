# Claude Deployment Guide

This repository contains the **Sugarcane Intelligent Analysis System**, which consists of two AI prediction modules. This file is specifically designed to help AI assistants (like Claude) deploy, debug, and run this project automatically.

## 1. Project Architecture

The project has two independent sub-modules that share a Python environment.

### Module 1: `pytorch-ganzhe/`
- **Purpose**: Predict sugarcane growth height using an Attention-LSTM time series model.
- **Backend**: FastAPI serving the model on port **8002**.
- **Frontend**: Streamlit interface running on port **8501**, which communicates with the backend.
- **Weights**: Model weights are built into the directory (`best_attention_lstm.pth`). Training is NOT required.

### Module 2: `ultralytics-main1/ultralytics-main/`
- **Purpose**: Classify sugarcane leaf diseases using Computer Vision (YOLO11).
- **Backend 1**: A FastAPI endpoint for processing **Images**, running on port **8000** (`picture_api.py`).
- **Backend 2**: A FastAPI endpoint for processing **Videos**, running on port **8001** (`video_api.py`).
- **Weights**: Model weights are located at `runs/classify/sugarcane_cls_v1_optimized/weights/best.pt`. Training is NOT required.

---

## 2. Automated Deployment Steps

When the user asks you to deploy or start the project, follow these exact steps:

### Step 1: Python Environment
Preferably use `conda` to create an isolated Python 3.10 environment to prevent package conflicts.
```bash
conda create -n sugarcane python=3.10 -y
conda activate sugarcane
```

### Step 2: Install PyTorch (GPU Recommended)
Install PyTorch compatible with the host's CUDA version (e.g., cu118):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Dependencies
```bash
pip install -r pytorch-ganzhe/requirements.txt
pip install -r ultralytics-main1/ultralytics-main/requirements.txt
```

### Step 4: Run the Services
**Option A: VS Code One-Click Run**
If the user is using VS Code, recommend opening the `Run and Debug` tab and selecting `🚀 一键运行所有服务 (Run All)`. This utilizes the `.vscode/launch.json` configuration we have already provided.

**Option B: Manual Terminal Execution**
If you need to start the services in the terminal, start them as background jobs or in separate terminal instances:

```bash
# 1. Height Prediction Backend (8002)
cd pytorch-ganzhe && uvicorn lstm_api:app --host 0.0.0.0 --port 8002 --reload

# 2. Height Prediction Frontend (8501)
cd pytorch-ganzhe/web && python start.py

# 3. Disease Image Detection API (8000)
cd ultralytics-main1/ultralytics-main && uvicorn picture_api:app --host 0.0.0.0 --port 8000 --reload

# 4. Disease Video Detection API (8001)
cd ultralytics-main1/ultralytics-main && uvicorn video_api:app --host 0.0.0.0 --port 8001 --reload
```

---

## 3. Retraining the Models (If Requested)

Models are pre-trained. If the user explicitly asks you to retrain them on new datasets:

- **Module 1 (LSTM)**: Run `python main.py` inside `pytorch-ganzhe/`. The dataset is automatically read from `web/sugar_cane_growth_data.csv`.
- **Module 2 (YOLO11)**: Download `yolo11s-cls.pt` to `ultralytics-main1/ultralytics-main`, modify the `data_dir` in `sugar_train.py`, and run `python sugar_train.py`.

## 4. Troubleshooting
- **Missing `best.pt`**: Ensure the relative path in `picture_api.py` and `video_api.py` correctly points to the `runs/.../best.pt` file.
- **Port Conflicts**: Ensure ports `8000`, `8001`, `8002`, and `8501` are not occupied by other applications on the user's host machine.
