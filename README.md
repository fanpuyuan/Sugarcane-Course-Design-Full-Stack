# 🌿 甘蔗智能分析系统 | Sugarcane Intelligent Analysis System

> 本项目包含两个独立的子模块，分别用于甘蔗**叶片病害识别**（计算机视觉）和甘蔗**植株高度生长预测**（时间序列预测）。

两个子模块：
- **模块一** `pytorch-ganzhe/` — 基于 Attention-LSTM 的甘蔗**株高预测** + Streamlit 前端
- **模块二** `ultralytics-main1/` — 基于 YOLO11 的甘蔗**病害识别** API

> 💡 **AI 助手提示**：本项目包含 [CLAUDE.md](CLAUDE.md) 指南。如果您是 AI 助手（如 Claude），请优先阅读该文档以获取自动化部署和运行指令。

> ⚠️ **许可证说明**：本项目的 YOLO11 功能依赖 [Ultralytics](https://github.com/ultralytics/ultralytics)，该库采用 **AGPL-3.0** 协议。若用于商业用途，需向 Ultralytics 购买商业许可。本项目仅上传自行编写的代码，不含 Ultralytics 完整源码。

---

## 1. 环境准备 (Environment Setup)

建议使用 Conda 创建全新的虚拟环境，并安装对应你电脑显卡驱动版本的 PyTorch（以支持 GPU 加速跑 YOLO11）。

```bash
# 1. 创建并激活一个 Python 3.10 环境
conda create -n sugarcane python=3.10
conda activate sugarcane

# 2. 安装 PyTorch (以 CUDA 11.8 为例，请改写为你支持的 CUDA 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装模块一依赖
pip install -r pytorch-ganzhe/requirements.txt

# 4. 安装模块二依赖（含 ultralytics）
pip install -r ultralytics-main1/ultralytics-main/requirements.txt
```

---

## 2. 快速启动服务 (Quick Start Services)

在已有模型权重的情况下，可直接按以下命令启动各项分析服务（首次部署若无模型，请先查看第 3 节进行训练）。

### ✨ 推荐：VS Code 一键启动 (One-Click Start)
本项目已配置好 `.vscode/launch.json`。如果您使用 VS Code：
1. 请先在左下角选择您刚创建的 `sugarcane` Conda 环境对应的 Python 解释器。
2. 点击左侧活动栏的 **运行和调试 (Run and Debug)** 图标（或快捷键 `Ctrl+Shift+D`）。
3. 在顶部的下拉菜单中选择 **"🚀 一键运行所有服务 (Run All)"**。
4. 点击绿色的播放按钮，即可一次性全部启动前端和三个后端 API 服务！

### 手动独立启动方式
建议新开若干个终端并激活上面创建的 conda 环境后再分别运行服务：

### 端口与服务汇总
| 服务模块 | 服务功能 | 端口 | 本地访问地址 |
|------|------|------|-------------|
| 模块一 | Streamlit 前端页面 | **8501** | http://localhost:8501 |
| 模块一 | 株高预测后端 API | **8002** | http://localhost:8002/docs |
| 模块二 | 图片病害检测 API | **8000** | http://localhost:8000/docs |
| 模块二 | 视频病害处理 API | **8001** | http://localhost:8001/docs |

### 启动模块一：株高预测服务
**终端 1 — 预测后端 API**
```bash
cd pytorch-ganzhe
uvicorn lstm_api:app --host 0.0.0.0 --port 8002 --reload
```
**终端 2 — Streamlit 前端**
```bash
cd pytorch-ganzhe/web
python start.py
```

### 启动模块二：病害识别服务
**终端 1 — 图片识别 API**
```bash
cd ultralytics-main1/ultralytics-main
uvicorn picture_api:app --host 0.0.0.0 --port 8000 --reload
```
**终端 2 — 视频处理 API**
```bash
cd ultralytics-main1/ultralytics-main
uvicorn video_api:app --host 0.0.0.0 --port 8001 --reload
```

---

## 3. 自行训练模型 (Training Models)

> **💡 温馨提示：本项目已在仓库中内置了预训练好的最佳模型权重，开箱即用，无需您亲自训练即可启动服务。** 如果您希望更换数据集或自己尝试训练过程，请参考以下说明：

### 模块一：株高预测（Attention-LSTM）的训练
数据将被默认从 `pytorch-ganzhe/web/sugar_cane_growth_data.csv` 读取：
```bash
cd pytorch-ganzhe
python main.py
```
训练结束后，将在 `pytorch-ganzhe/` 目录下自动生成：
- `best_attention_lstm.pth` — 模型权重文件
- `models/scaler_X.pkl`、`models/scaler_y.pkl` — 数据归一化参数
- `prediction.png` — 实际与预测对比可视化结果

### 模块二：病害识别（YOLO11）的训练
1. 请自行下载 YOLO11 官方初始化权重 [yolo11s-cls.pt](https://github.com/ultralytics/assets/releases) 并置于 `ultralytics-main1/ultralytics-main` 目录。
2. 打开 `sugar_train.py`，将第 6 行的 `data_dir` 修改为您本地甘蔗病害叶片数据集路径（需组织为标准 `ImageFolder` 格式，即直接按照类别建侧子文件夹存放图像）。
3. 执行训练：
```bash
cd ultralytics-main1/ultralytics-main
python sugar_train.py
```
训练出的最佳模型将会默认保存在：`runs/classify/sugarcane_cls_v1_optimized/weights/best.pt`

---

## 4. License

本项目所含自行编写的代码采用 **MIT License**。  
请注意，项目的叶面病害推理功能在底层依赖了 [Ultralytics](https://github.com/ultralytics/ultralytics) 框架，该框架采用 **AGPL-3.0** 开源许可，因此若您有闭源或商业使用计划，请先主动向原作者获准[商业许可](https://www.ultralytics.com/license)。
