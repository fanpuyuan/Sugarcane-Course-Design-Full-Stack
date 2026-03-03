# 文件名: prediction_utils.py
import os
import joblib
import pandas as pd
import numpy as np
import torch
from Net import AttentionLSTM  # 假设你的模型定义在 Net.py 中

# --- 配置 ---
MODEL_PATH = "best_attention_lstm.pth"  # 你训练好的模型文件路径
SCALER_X_PATH = "models/scaler_X.pkl"   # 特征标准化器路径
SCALER_Y_PATH = "models/scaler_y.pkl"   # 目标值标准化器路径
INPUT_SEQ_LENGTH = 45  # 输入序列长度
OUTPUT_HORIZON = 15    # 预测时间范围
INPUT_FEATURES = 6     # 输入特征数量 (不包括目标变量本身)

def load_model_and_scalers():
    """加载训练好的模型和标准化器"""
    print("正在加载模型和标准化器...")
    try:
        # 加载模型
        model = AttentionLSTM(
            input_dim=INPUT_FEATURES + 1,  # 6个特征 + 1个目标变量 (因为预测时目标变量也在序列中)
            hidden_dim=64,
            output_horizon=OUTPUT_HORIZON,
            dropout=0.2
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        print("✅ 模型加载成功")

        # 加载标准化器
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        print("✅ 标准化器加载成功")

        return model, scaler_X, scaler_y
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ 加载模型或标准化器时出错: {e}")
        return None, None, None

def prepare_input_data(df, scaler_X, scaler_y, target_col_name='plant_height'):
    """
    准备预测所需的输入数据
    :param df: 包含最近数据的DataFrame
    :param scaler_X: 特征标准化器
    :param scaler_y: 目标值标准化器
    :param target_col_name: 目标列名
    :return: 标准化后的输入序列 (1, 45, 7) 或 None
    """
    try:
        # 检查必要的列是否存在
        required_cols = ['temperature', 'precipitation', 'sunshine_hours', 'soil_moisture', 'leaf_area_index', 'stem_diameter', target_col_name]
        if not all(col in df.columns for col in required_cols):
            print(f"❌ 输入数据缺少必要列: {required_cols}")
            return None

        # 提取特征和目标值
        feature_cols = [col for col in required_cols if col != target_col_name]
        X_data = df[feature_cols].values
        y_data = df[[target_col_name]].values

        # 检查数据长度
        if X_data.shape[0] < INPUT_SEQ_LENGTH:
            print(f"❌ 输入数据长度不足，需要至少 {INPUT_SEQ_LENGTH} 天的数据，当前只有 {X_data.shape[0]} 天。")
            return None

        # 标准化
        X_scaled = scaler_X.transform(X_data[-INPUT_SEQ_LENGTH:]) # 取最后45天的特征
        y_scaled = scaler_y.transform(y_data[-INPUT_SEQ_LENGTH:]) # 取最后45天的目标值

        # 合并为完整序列 (45, 7)
        scaled_seq = np.concatenate([X_scaled, y_scaled], axis=1)

        # 转换为 PyTorch Tensor (1, 45, 7) - batch_size=1
        input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0)

        return input_tensor
    except Exception as e:
        print(f"❌ 准备输入数据时出错: {e}")
        return None

def run_prediction(model, input_tensor):
    """
    执行预测
    :param model: 加载好的PyTorch模型
    :param input_tensor: 标准化后的输入Tensor
    :return: 反归一化后的预测结果 (15,) 或 None
    """
    try:
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy().flatten() # Shape: (15,)
        return pred_scaled
    except Exception as e:
        print(f"❌ 模型预测时出错: {e}")
        return None

def inverse_transform_prediction(pred_scaled, scaler_y):
    """
    将预测结果反归一化
    :param pred_scaled: 标准化后的预测结果
    :param scaler_y: 目标值标准化器
    :return: 反归一化后的预测结果 (15,)
    """
    try:
        pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten() # Shape: (15,)
        return pred_original
    except Exception as e:
        print(f"❌ 反归一化预测结果时出错: {e}")
        return None

# --- 预测主函数 ---
def predict_future_height(input_data_df):
    """
    预测未来15天的植株高度
    :param input_data_df: 包含最近45天历史数据的DataFrame
    :return: 包含预测结果的字典 {'success': bool, 'predictions': list, 'error': str}
    """
    model, scaler_X, scaler_y = load_model_and_scalers()
    if model is None or scaler_X is None or scaler_y is None:
        return {"success": False, "predictions": [], "error": "模型或标准化器加载失败"}

    input_tensor = prepare_input_data(input_data_df, scaler_X, scaler_y)
    if input_tensor is None:
        return {"success": False, "predictions": [], "error": "输入数据准备失败"}

    pred_scaled = run_prediction(model, input_tensor)
    if pred_scaled is None:
        return {"success": False, "predictions": [], "error": "模型预测失败"}

    predictions = inverse_transform_prediction(pred_scaled, scaler_y)
    if predictions is None:
        return {"success": False, "predictions": [], "error": "预测结果反归一化失败"}

    return {"success": True, "predictions": predictions.tolist(), "error": ""}

# --- 示例用法 (如果直接运行此脚本) ---
if __name__ == "__main__":
    # 模拟输入数据 (实际使用时，这个DataFrame应该来自API请求)
    # 假设你有一个包含最近45天数据的CSV文件 'recent_45_days.csv'
    # recent_data_df = pd.read_csv('recent_45_days.csv')
    # recent_data_df['date'] = pd.to_datetime(recent_data_df['date']) # 如果有日期列

    # # 或者手动创建一个示例数据框 (确保列名和顺序正确)
    # sample_data = {
    #     'temperature': np.random.rand(45) * 30 + 15,  # Example range
    #     'precipitation': np.random.rand(45) * 10,
    #     'sunshine_hours': np.random.rand(45) * 12,
    #     'soil_moisture': np.random.rand(45) * 100,
    #     'plant_height': np.random.rand(45) * 200 + 50, # Example range
    #     'leaf_area_index': np.random.rand(45) * 10,
    #     'stem_diameter': np.random.rand(45) * 5,
    # }
    # recent_data_df = pd.DataFrame(sample_data)
    #
    # result = predict_future_height(recent_data_df)
    # print(result)
    pass # 如果不直接运行此脚本，pass 即可