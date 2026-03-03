# 文件名: yolo_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
from PIL import Image
import io
import torch
from ultralytics import YOLO
import os
import tempfile
import uuid

app = FastAPI(title="甘蔗病虫害YOLO分类API")

# --- 配置 ---
model_path = "runs/classify/sugarcane_cls_v1_optimized/weights/best.pt"
confidence_threshold = 0.3
input_size = 320  # 与你的模型训练时的输入尺寸保持一致

# 检查模型文件是否存在
if not os.path.exists(model_path):
    raise RuntimeError(f"模型文件不存在: {model_path}")

# 加载模型
print(f"✅ 加载模型: {model_path}")
model = YOLO(model_path)

# 检查 CUDA 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")
model.to(device)
model.eval()  # 设置为评估模式

@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    """
    上传图片并进行分类预测
    """
    try:
        # 1. 读取上传的图片文件
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="上传的文件不是有效的图片格式")

        # 2. 使用 YOLO 进行预测
        results = model.predict(
            source=image,
            verbose=False,
            conf=confidence_threshold,
            imgsz=input_size,
            device=device
        )

        # 3. 解析预测结果
        result = results[0]
        if result.probs is not None:
            top1_class_idx = result.probs.top1
            top1_confidence = result.probs.top1conf.item()
            class_names = result.names
            predicted_class_name = class_names[top1_class_idx]

            # 返回预测结果
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "prediction": {
                        "class": predicted_class_name,
                        "confidence": round(top1_confidence, 4)
                    }
                }
            )
        else:
            # 没有检测到任何类别
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "prediction": {
                        "class": "Unknown",
                        "confidence": 0.0
                    }
                }
            )

    except Exception as e:
        # 处理预测过程中的错误
        print(f"预测过程中出错: {e}")
        raise HTTPException(status_code=500, detail=f"预测过程中出错: {str(e)}")

@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    """
    上传视频文件并进行分类预测，返回处理后的视频文件。
    注意：这是一个简化的实现，它会处理整个视频并返回一个新视频。
    """
    try:
        # 1. 将上传的视频文件保存到临时位置
        temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{uuid.uuid4()}.mp4")
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await file.read())

        # 2. 打开视频文件
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="无法打开上传的视频文件")

        # 3. 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 4. 创建输出视频文件
        temp_output_path = os.path.join(tempfile.gettempdir(), f"output_video_{uuid.uuid4()}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}...") # 可选的进度打印

            # 5. YOLO 预测
            results = model.predict(
                source=frame,
                verbose=False,
                conf=confidence_threshold,
                imgsz=input_size,
                device=device
            )

            # 6. 解析预测结果并绘制
            result = results[0]
            if result.probs is not None:
                top1_class_idx = result.probs.top1
                top1_confidence = result.probs.top1conf.item()
                class_names = result.names
                predicted_class_name = class_names[top1_class_idx]

                # 在帧上绘制预测结果
                label_text = f"{predicted_class_name}: {top1_confidence:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (10, 10), (10 + text_width, 10 + text_height + baseline), (0, 255, 0), -1)
                cv2.putText(frame, label_text, (10, 10 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 7. 写入输出视频
            out.write(frame)

        # 8. 释放资源
        cap.release()
        out.release()

        # 9. 删除临时输入视频
        os.remove(temp_video_path)

        # 10. 返回处理后的视频文件
        def iterfile():
            with open(temp_output_path, 'rb') as f:
                yield from f
            # 删除临时输出视频
            os.remove(temp_output_path)

        return StreamingResponse(iterfile(), media_type="video/mp4", headers={"Content-Disposition": f"attachment; filename=output.mp4"})

    except Exception as e:
        # 处理预测过程中的错误
        print(f"视频预测过程中出错: {e}")
        # 尝试删除临时文件
        try:
            os.remove(temp_video_path)
        except:
            pass
        try:
            os.remove(temp_output_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"视频预测过程中出错: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "甘蔗病虫害YOLO分类API服务运行中"}

if __name__ == "__main__":
    import uvicorn
    # 启动服务，监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)