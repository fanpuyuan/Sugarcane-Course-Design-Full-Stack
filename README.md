# 甘蔗智能分析系统

两个子模块：
- **模块一** `pytorch-ganzhe/` — 基于 Attention-LSTM 的甘蔗**株高预测** + Streamlit 前端
- **模块二** `ultralytics-main1/` — 基于 YOLO11 的甘蔗**病害识别** API

> ⚠️ **许可证说明**：本项目的 YOLO11 功能依赖 [Ultralytics](https://github.com/ultralytics/ultralytics)，该库采用 **AGPL-3.0** 协议。若用于商业用途，需向 Ultralytics 购买商业许可。本项目仅上传自行编写的代码，不含 Ultralytics 完整源码。

---

## 环境准备

使用 conda 环境 `yolodemo01221`（已含 torch 2.5.1）。

```bash
# 模块一依赖
pip install -r pytorch-ganzhe/requirements.txt

# 模块二依赖（含 ultralytics）
pip install -r ultralytics-main1/ultralytics-main/requirements.txt
```

---

## 模块一：株高预测（Attention-LSTM）

> 目录：`pytorch-ganzhe/`

### 训练模型（首次使用才需要跑）

数据文件默认路径：`pytorch-ganzhe/web/sugar_cane_growth_data.csv`

```bash
cd pytorch-ganzhe
python main.py
```

训练后生成：
- `best_attention_lstm.pth` — 模型权重（不上传 GitHub）
- `models/scaler_X.pkl`、`models/scaler_y.pkl` — 归一化参数（不上传 GitHub）
- `prediction.png` — 预测结果图

### 启动服务（两个终端分别运行）

**终端 1 — 预测后端 API（端口 8002）**
```bash
cd pytorch-ganzhe
uvicorn lstm_api:app --host 0.0.0.0 --port 8002 --reload
```

**终端 2 — Streamlit 前端**
```bash
cd pytorch-ganzhe/web
python start.py
```

前端地址：http://localhost:8501  
API 文档：http://localhost:8002/docs

---

## 模块二：病害识别（YOLO11）

> 目录：`ultralytics-main1/ultralytics-main/`

### 训练模型（首次使用才需要跑）

1. 下载预训练权重 [yolo11s-cls.pt](https://github.com/ultralytics/assets/releases) 放入该目录
2. 修改 `sugar_train.py` 第 6 行 `data_dir` 为你的数据集路径（ImageFolder 格式）
3. 运行训练：

```bash
cd ultralytics-main1/ultralytics-main
python sugar_train.py
```

训练好的模型保存在 `runs/classify/sugarcane_cls_v1_optimized/weights/best.pt`

### 启动服务（两个终端分别运行）

**终端 1 — 图片识别 API（端口 8000）**
```bash
cd ultralytics-main1/ultralytics-main
uvicorn picture_api:app --host 0.0.0.0 --port 8000 --reload
```

**终端 2 — 视频处理 API（端口 8001）**
```bash
cd ultralytics-main1/ultralytics-main
uvicorn video_api:app --host 0.0.0.0 --port 8001 --reload
```

图片 API 文档：http://localhost:8000/docs  
视频 API 文档：http://localhost:8001/docs

---

## 端口汇总

| 服务 | 端口 |
|------|------|
| Streamlit 前端 | 8501 |
| 株高预测 API | 8002 |
| 图片病害识别 API | 8000 |
| 视频病害处理 API | 8001 |

---

## 上传 GitHub 后别人如何使用

1. `git clone` 本项目
2. 按上方「环境准备」安装依赖
3. 自行训练模型（或联系作者获取权重文件）
4. 按模块启动对应服务

---

## License

本项目自编代码采用 MIT 协议。  
YOLO11 推理功能依赖 Ultralytics（AGPL-3.0），商业使用需获得其[商业许可](https://www.ultralytics.com/license)。
