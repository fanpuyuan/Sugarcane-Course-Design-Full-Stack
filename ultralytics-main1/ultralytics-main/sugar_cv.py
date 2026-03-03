from ultralytics import YOLO
import cv2
import torch
import os

# --- 配置 ---
model_path = "runs/classify/sugarcane_cls_v1_optimized/weights/best.pt"
video_path = "sugarcanemp4.mp4"
video_source = video_path # 0 表示默认摄像头，也可以是视频文件路径，如 "path/to/your/video.mp4"
confidence_threshold = 0.3  # 预测置信度阈值，低于此值不显示
# --- 配置结束 ---

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"❌ 模型文件不存在: {model_path}")
    exit()

# 加载模型
print(f"✅ 加载模型: {model_path}")
model = YOLO(model_path)

# 检查 CUDA 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")
model.to(device)  # 将模型移动到指定设备

# 打开视频源 (摄像头或视频文件)
print(f"✅ 打开视频源: {video_source}")
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"❌ 无法打开视频源: {video_source}")
    exit()

print("✅ 视频流已打开，按 'q' 退出")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取视频帧失败，可能已到达视频末尾或摄像头断开。")
            break

        # YOLOv8-cls 预测
        # verbose=False 关闭预测过程的日志输出
        # conf=confidence_threshold 设置置信度阈值
        # imgsz=320 使用与训练时一致的输入尺寸 (如果训练时改了，这里也要改)
        results = model.predict(
            source=frame,
            verbose=False,
            conf=confidence_threshold,
            imgsz=320, # 请确保这个尺寸与您训练时使用的尺寸一致
            device=device
        )

        # 解析预测结果
        result = results[0] # 获取第一个结果 (当前帧)
        if result.probs is not None: # 确保有预测概率
            # 获取最高概率的类别索引和置信度
            top1_class_idx = result.probs.top1
            top1_confidence = result.probs.top1conf.item() # 转换为 Python 标量

            # 获取类别名称 (确保训练时类别顺序与此处一致)
            class_names = result.names # {0: 'Healthy', 1: 'Mosaic', ...}
            predicted_class_name = class_names[top1_class_idx]

            # 在帧上绘制预测结果
            label_text = f"{predicted_class_name}: {top1_confidence:.2f}"
            # 计算文本框大小
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # 绘制背景矩形
            cv2.rectangle(frame, (10, 10), (10 + text_width, 10 + text_height + baseline), (0, 255, 0), -1)
            # 绘制文本
            cv2.putText(frame, label_text, (10, 10 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # (可选) 绘制置信度条
            bar_length = 200
            bar_height = 15
            bar_x_start = 10
            bar_y_start = 30 + text_height + baseline
            filled_length = int((top1_confidence / 1.0) * bar_length)
            cv2.rectangle(frame, (bar_x_start, bar_y_start), (bar_x_start + bar_length, bar_y_start + bar_height), (255, 255, 255), -1)
            cv2.rectangle(frame, (bar_x_start, bar_y_start), (bar_x_start + filled_length, bar_y_start + bar_height), (0, 255, 0), -1)
            cv2.putText(frame, f"{top1_confidence:.2f}", (bar_x_start + bar_length + 5, bar_y_start + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 显示处理后的帧
        cv2.imshow('Sugarcane Disease Classification - Live Feed', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 用户按下 'q' 键，退出视频流分类。")
            break

except KeyboardInterrupt:
    print("\n👋 用户中断程序 (Ctrl+C)，退出视频流分类。")

finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 摄像头/视频流已关闭，OpenCV窗口已销毁。")

print("🎉 视频流分类程序结束。")