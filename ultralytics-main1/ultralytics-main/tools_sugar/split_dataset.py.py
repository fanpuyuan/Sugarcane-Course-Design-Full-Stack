import os
import shutil
import random
from pathlib import Path

# ================== 配置参数 ==================
original_root = r"C:\Users\33985\Downloads\archive"  # 原始数据集路径
output_root = r"C:\Users\33985\Downloads\archive_classify"  # 输出路径
train_ratio = 0.8  # 训练集比例
random_seed = 42  # 随机种子，确保每次划分一致
# ==============================================

# 设置随机种子
random.seed(random_seed)

# 创建输出目录
os.makedirs(output_root, exist_ok=True)
train_dir = os.path.join(output_root, "train")
val_dir = os.path.join(output_root, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取所有类别文件夹
class_dirs = [d for d in os.listdir(original_root) if os.path.isdir(os.path.join(original_root, d))]

print(f"✅ 发现 {len(class_dirs)} 个类别：{class_dirs}")

for class_name in class_dirs:
    class_path = os.path.join(original_root, class_name)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if len(image_files) == 0:
        print(f"⚠️  类别 '{class_name}' 中没有图片，跳过")
        continue

    # 打乱顺序
    random.shuffle(image_files)

    # 划分训练集和验证集
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # 创建目标子目录
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # 复制文件
    for img in train_files:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy2(src, dst)

    for img in val_files:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_dir, img)
        shutil.copy2(src, dst)

    print(f"📁 {class_name}: {len(train_files)} 训练 + {len(val_files)} 验证")

print(f"\n🎉 数据集划分完成！")
print(f"训练集路径: {train_dir}")
print(f"验证集路径: {val_dir}")