import os

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from Net import create_dataloaders,AttentionLSTM,mape_loss,train_model
from tools import plot_original_data

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. 数据准备
def data_load(path):
    # 读取数据（自动解析日期）
    df = pd.read_csv(path, parse_dates=['date'])
    print(f"原始数据形状: {df.shape}")
    print(f"缺失值统计:\n{df.isnull().sum()}")
    return df


def fill_missing_agri(df):
    df = df.copy()

    # 1. 气象三要素：线性插值 + 边界填充
    meteo_cols = ['temperature', 'precipitation', 'sunshine_hours']
    df[meteo_cols] = df[meteo_cols].interpolate(method='linear', limit_direction='both')
    df[meteo_cols] = df[meteo_cols].ffill().bfill()

    # 2. 土壤墒情
    if df['soil_moisture'].notna().sum() >= 5:
        df['soil_moisture'] = df['soil_moisture'].interpolate(
            method='spline', order=3, limit_direction='both'
        )
    df['soil_moisture'] = df['soil_moisture'].fillna(df['soil_moisture'].median())

    # 3. 作物生长指标
    crop_cols = ['plant_height', 'leaf_area_index', 'stem_diameter']
    for col in crop_cols:
        # 滑动均值填充
        df[col] = df[col].fillna(
            df[col].rolling(window=7, min_periods=1, center=True).mean()
        )
        df[col] = df[col].ffill().bfill()

        # 生物学约束
        if col in ['plant_height', 'stem_diameter']:
            df[col] = df[col].cummax()

    print("填充后缺失值总数:", df.isnull().sum().sum())
    return df


def data_normalization(df, target_col_name='plant_height'):
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    # 验证是否还有非数值列
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] != df.shape[1]:
        print("警告：存在非数值列，已自动过滤")
        df = numeric_df

    # 分离特征和目标
    feature_cols = [col for col in df.columns if col != target_col_name]
    target_col = target_col_name

    X_data = df[feature_cols].values  # (180, 6)
    y_data = df[[target_col]].values  # (180, 1) —— 注意保持2D

    # 创建两个独立的 scaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)

    # 合并为完整输入序列（7列）
    scaled_data = np.concatenate([X_scaled, y_scaled], axis=1)  # (180, 7)
    # 在 data_normalization 返回前打印列顺序
    print("归一化后列顺序:", feature_cols + ['plant_height'])
    # 输出应为：
    # ['temperature', 'precipitation', 'sunshine_hours',
    #  'soil_moisture', 'leaf_area_index', 'stem_diameter', 'plant_height']

    # 返回完整数据 + 两个 scaler
    return scaled_data, scaler_X, scaler_y


# 2. 构建滑动窗口
def create_sequences(data, lookback=45, horizon=15, target_col=4):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, target_col])
    return np.array(X), np.array(y)


# 主程序
if __name__ == '__main__':
    path = 'web/sugar_cane_growth_data.csv'
    target_col_name = 'plant_height'
    seq_length = 45  # 建议45天（覆盖关键生长期）
    pred_length = 15  # 预测未来15天
    target_col = 6  # plant_height 在数值列中的位置（0:temp, 1:precip, 2:sun, 3:soil, 4:height, 5:LAI, 6:stem），但最终是最后一个

    # 加载并处理数据
    data = data_load(path)
    data_filled = fill_missing_agri(data)

    plot_original_data(data)

    # 归一化（自动处理 date 列）
    scaled_data,  scaler_X, scaler_y= data_normalization(data_filled,target_col_name)

    print(f"归一化后数据形状: {scaled_data.shape}")  # 应为 (180, 7)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    print("✅ Scaler 已保存至 models/ 目录")

    # 构建序列
    X, y = create_sequences(scaled_data, lookback=seq_length, horizon=pred_length, target_col=target_col)
    print(f"序列构建完成: X={X.shape}, y={y.shape}")


    train_loader, val_loader, test_loader, (X_test, y_test) = create_dataloaders(X, y)

    model = AttentionLSTM(
        input_dim=7,  # 7个特征
        hidden_dim=64,
        output_horizon=15,
        dropout=0.2
    )
    print("模型结构:")
    print(model)

    trained_model, history = train_model(
        model, train_loader, val_loader, device, epochs=200, patience=15
    )

    #测试
    trained_model.eval()
    with torch.no_grad():
        y_pred_test = trained_model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    # 反归一化（仅对目标变量）
    y_test_inv = scaler_y.inverse_transform(y_test)# (N, 15)
    print(y_test_inv)
    y_pred_inv = scaler_y.inverse_transform(y_pred_test)  # (N, 15)

    # 计算最终 MAPE
    final_mape = np.mean(np.abs((y_test_inv - y_pred_inv) / np.clip(np.abs(y_test_inv), 1e-6, None))) * 100
    print(f"\n✅ 测试集 MAPE: {final_mape:.2f}%")

    # 可视化
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 4))
    plt.plot(y_test_inv[0], 'o-', label='真实值')
    plt.plot(y_pred_inv[0], 's--', label='预测值')
    plt.title(f'未来15天株高预测 (MAPE={final_mape:.2f}%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction.png', dpi=150)
    plt.show()