from ultralytics import YOLO
import torch
import os

# ================== 配置参数 ==================
data_dir = r"C:\Users\33985\Downloads\archive_classify"  #
model_name = "yolo11s-cls.pt"  # 预训练分类模型
epochs = 5  # 训练轮数
imgsz = 320  # 输入图像尺寸 (宽, 高)
batch = 16   # 批大小 (根据显存调整，如果显存不够可以改为8)
device = "0" if torch.cuda.is_available() else "cpu"  # GPU 或 CPU
name = "sugarcane_cls_v1_optimized"
# ==============================================
if __name__ == '__main__':
    print(f"✅ 使用设备: {device}")
    print(f"✅ 图像尺寸: {imgsz}x{imgsz}")
    print(f"✅ 批大小: {batch}")
    print(f"✅ 数据集路径: {data_dir}")

    # 加载预训练分类模型
    model = YOLO(model_name)

    # 开始训练
    results = model.train(
        data=data_dir,    # 数据集路径
        epochs=epochs,    # 训练轮数
        imgsz=imgsz,      # 图像尺寸 (YOLOv8-cls 会自动调整)
        batch=batch,      # 批大小
        device=device,    # 设备
        name=name,        # 任务名称 (结果保存在 runs/classify/name)
        patience=10,      # 早停轮数，如果10轮内验证指标没有提升则停止
        save_period=5,    # 每5轮保存一次模型
        plots=True,       # 生成训练结果图
        optimizer='Adam', # 优化器
        lr0=0.001,        # 初始学习率
        lrf=0.01,         # 最终学习率 (余弦退火)
        warmup_epochs=3,  # 预热轮数
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # 数据增强
        degrees=10.0,     # 随机旋转角度
        translate=0.1,    # 随机平移比例
        scale=0.2,        # 随机缩放比例
        shear=0.0,        # 随机剪切角度
        perspective=0.0,  # 随机透视变换
        flipud=0.0,       # 上下翻转概率 (甘蔗叶片通常有方向性，可设为0)
        fliplr=0.5,       # 左右翻转概率
        mosaic=0.0,       # 马赛克增强 (分类任务通常不用)
        mixup=0.0,        # MixUp 增强 (分类任务可选)
        copy_paste=0.0,   # Copy-Paste 增强 (分类任务不用)
        # Val
        val=True,         # 是否验证
        split='val',      # 验证集划分
        save=True,        # 是否保存模型
        exist_ok=True,    # 是否覆盖已有任务
    )

    print("🎉 训练完成！")
    print(f"📊 训练日志和模型保存在: runs/classify/{name}/")
    print(f"🎯 最佳模型路径: runs/classify/{name}/weights/best.pt")

    metrics = model.val()
    print("验证集指标:", metrics)