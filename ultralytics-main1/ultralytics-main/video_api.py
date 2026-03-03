# 文件名: yolo_video_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import tempfile
import uuid
import threading
import time
from queue import Queue
import asyncio
import aiofiles

app = FastAPI(title="甘蔗病虫害YOLO视频检测API")

# --- 配置 ---
model_path = "runs/classify/sugarcane_cls_v1_optimized/weights/best.pt"
confidence_threshold = 0.3
input_size = 320

# --- 静态文件目录 ---
STATIC_DIR = "processed_videos"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- 任务管理 ---
# 简单的内存存储，实际生产应使用数据库或Redis
tasks = {}
task_queue = Queue()

# 加载模型
if not os.path.exists(model_path):
    raise RuntimeError(f"模型文件不存在: {model_path}")
model = YOLO(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


class VideoProcessor:
    def __init__(self, task_id, video_source, is_rtsp=False):
        self.task_id = task_id
        self.video_source = video_source
        self.is_rtsp = is_rtsp
        self.status = "processing"
        self.progress = 0
        self.results = []  # 存储关键帧的检测结果
        self.error = None
        self.output_video_filename = f"output_{task_id}.mp4"
        self.output_video_path = os.path.join(STATIC_DIR, self.output_video_filename)

    def process(self):
        try:
            if self.is_rtsp:
                cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            else:
                temp_video_path = self.video_source
                cap = cv2.VideoCapture(temp_video_path)

            if not cap.isOpened():
                self.error = "无法打开视频源"
                self.status = "failed"
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 创建输出视频文件
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))

            frame_count = 0
            results_summary = {"healthy": 0, "diseased": 0, "unknown": 0}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                self.progress = min(99, int((frame_count / total_frames) * 100))

                results = model.predict(
                    source=frame,
                    verbose=False,
                    conf=confidence_threshold,
                    imgsz=input_size,
                    device=device
                )

                result = results[0]
                if result.probs is not None:
                    top1_class_idx = result.probs.top1
                    top1_confidence = result.probs.top1conf.item()
                    class_names = result.names
                    predicted_class_name = class_names[top1_class_idx]

                    if "healthy" in predicted_class_name.lower():
                        results_summary["healthy"] += 1
                    elif "disease" in predicted_class_name.lower() or "pest" in predicted_class_name.lower():
                        results_summary["diseased"] += 1
                    else:
                        results_summary["unknown"] += 1

                    label_text = f"{predicted_class_name}: {top1_confidence:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (10, 10), (10 + text_width, 10 + text_height + baseline), (0, 255, 0), -1)
                    cv2.putText(frame, label_text, (10, 10 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                out.write(frame)

                if frame_count % int(fps) == 0:
                    self.results.append({
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps,
                        "prediction": {
                            "class": predicted_class_name,
                            "confidence": top1_confidence
                        }
                    })

            cap.release()
            out.release()
            self.progress = 100
            self.status = "completed"
            self.results.append({"summary": results_summary})

        except Exception as e:
            self.error = str(e)
            self.status = "failed"
            print(f"处理任务 {self.task_id} 时出错: {e}")


@app.post("/process_video/")
async def process_video_endpoint(file: UploadFile = File(None),
                                 rtsp_url: str = Query(None, description="RTSP视频流地址")):
    if file is None and rtsp_url is None:
        raise HTTPException(status_code=400, detail="必须提供视频文件或RTSP URL")

    task_id = str(uuid.uuid4())
    video_path_or_url = rtsp_url if rtsp_url else None

    if file:
        temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{task_id}.mp4")
        async with aiofiles.open(temp_video_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        video_path_or_url = temp_video_path

    processor = VideoProcessor(task_id, video_path_or_url, is_rtsp=(rtsp_url is not None))
    tasks[task_id] = processor

    thread = threading.Thread(target=processor.process)
    thread.start()

    return {"task_id": task_id, "status": "processing"}


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务未找到")

    task = tasks[task_id]
    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "results": task.results,
        "error": task.error,
        # 添加视频文件名，用于前端构建播放URL
        "output_video_filename": task.output_video_filename if task.status == "completed" else None
    }


# 可选：保留下载端点
@app.get("/download_video/{task_id}")
async def download_video(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务未找到")

    task = tasks[task_id]
    if task.status != "completed" or not os.path.exists(task.output_video_path):
        raise HTTPException(status_code=404, detail="处理后的视频不可用或任务未完成")

    return FileResponse(task.output_video_path, media_type="video/mp4", filename=task.output_video_filename)


@app.get("/")
def read_root():
    return {"message": "甘蔗病虫害YOLO视频检测API服务运行中"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)