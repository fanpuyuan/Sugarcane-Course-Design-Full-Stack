import pandas as pd
from Net import AttentionLSTM
import torch
# # 读取数据
# df = pd.read_csv('sugar_crop_data.csv')
# print(f"原始数据形状: {df.shape}")
# print(f"缺失值统计:\n{df.isnull().sum()}")

# 验证保存
# loaded_model = AttentionLSTM(input_dim=7, hidden_dim=64, output_horizon=15)
# loaded_model.load_state_dict(torch.load('best_attention_lstm.pth', weights_only=True))
# print("✅ 模型保存与加载验证通过！")

# backtest_custom.py
import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime
from Net import AttentionLSTM
from main import fill_missing_agri

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False


def load_model_and_scaler():
    model = AttentionLSTM(input_dim=7, hidden_dim=64, output_horizon=15)
    model.load_state_dict(torch.load('best_attention_lstm.pth', weights_only=True))
    model.eval()
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    return model, scaler_X, scaler_y


def preprocess_data(df, scaler_X, scaler_y):
    """预处理整个数据集"""
    if 'date' in df.columns:
        df = df.set_index('date')
    df = df.sort_index()

    feature_cols = [col for col in df.columns if col != 'plant_height']
    X_data = df[feature_cols].values
    y_data = df[['plant_height']].values

    X_scaled = scaler_X.transform(X_data)
    y_scaled = scaler_y.transform(y_data)
    scaled_data = np.concatenate([X_scaled, y_scaled], axis=1)

    return scaled_data, df.index


def find_date_range_indices(dates, start_date, end_date):
    """查找日期范围对应的索引"""
    start_idx = np.where(dates >= pd.Timestamp(start_date))[0]
    end_idx = np.where(dates <= pd.Timestamp(end_date))[0]

    if len(start_idx) == 0 or len(end_idx) == 0:
        raise ValueError(f"日期范围 {start_date} 至 {end_date} 不在数据中！")

    start_pos = start_idx[0]
    end_pos = end_idx[-1]

    if end_pos - start_pos + 1 != 15:
        print(f"⚠️ 警告: 指定范围有 {end_pos - start_pos + 1} 天，非15天")

    return start_pos, end_pos


def main():
    # ===== 用户配置区 =====
    TEST_START = '2023-8-13'  # 要预测的开始日期
    TEST_END = '2023-8-27'  # 要预测的结束日期
    # ======================

    # 1. 加载并处理数据
    df = pd.read_csv('web/sugar_cane_growth_data.csv', parse_dates=['date'])
    df_filled = fill_missing_agri(df)

    # 2. 加载模型
    model, scaler_X, scaler_y = load_model_and_scaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 3. 预处理
    scaled_data, dates = preprocess_data(df_filled, scaler_X, scaler_y)

    # 4. 确定测试范围索引
    test_start_idx, test_end_idx = find_date_range_indices(dates, TEST_START, TEST_END)
    horizon = test_end_idx - test_start_idx + 1

    # 5. 确定输入范围（预测前45天）
    input_end_idx = test_start_idx - 1
    input_start_idx = input_end_idx - 44  # 45天

    if input_start_idx < 0:
        raise ValueError(f"输入数据不足！需要从 {dates[input_start_idx]} 开始的数据")

    print(f"📅 测试配置:")
    print(
        f"- 输入数据: {dates[input_start_idx].strftime('%Y-%m-%d')} 至 {dates[input_end_idx].strftime('%Y-%m-%d')} ({45}天)")
    print(
        f"- 预测目标: {dates[test_start_idx].strftime('%Y-%m-%d')} 至 {dates[test_end_idx].strftime('%Y-%m-%d')} ({horizon}天)")

    # 6. 提取数据
    input_seq = scaled_data[input_start_idx:input_end_idx + 1]  # (45, 7)
    true_values_scaled = scaled_data[test_start_idx:test_end_idx + 1, -1]  # (15,)

    input_dates = dates[input_start_idx:input_end_idx + 1]
    true_dates = dates[test_start_idx:test_end_idx + 1]

    # 7. 预测
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy().flatten()[:horizon]  # 截取实际天数

    # 8. 反归一化
    true_values = scaler_y.inverse_transform(true_values_scaled.reshape(-1, 1)).flatten()
    pred_values = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    # 9. 计算 MAPE
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100

    mape = calculate_mape(true_values, pred_values)
    print(f"\n✅ 测试结果: MAPE = {mape:.2f}%")

    # 10. 绘制图表
    plt.figure(figsize=(14, 7))

    # === 修复：直接从 scaled_data 获取历史株高 ===
    plot_start_idx_global = max(0, input_start_idx - 30)
    plot_end_idx_global = test_end_idx

    # 提取历史株高（归一化值）
    historical_heights_scaled = scaled_data[plot_start_idx_global:plot_end_idx_global + 1, -1]
    historical_heights = scaler_y.inverse_transform(historical_heights_scaled.reshape(-1, 1)).flatten()
    plot_date_range = dates[plot_start_idx_global:plot_end_idx_global + 1]

    plt.plot(plot_date_range, historical_heights, 'o-', color='blue',
             label='历史观测值', linewidth=1.5, markersize=3)

    # 真实值和预测值（不变）
    plt.plot(true_dates, true_values, 's-', color='green', label='真实值', linewidth=2.5, markersize=6)
    plt.plot(true_dates, pred_values, 'D--', color='red', label='预测值', linewidth=2.5, markersize=6)

    # 标记预测起点
    plt.axvline(x=input_dates[-1], color='k', linestyle=':', alpha=0.7)
    plt.text(input_dates[-1], plt.ylim()[1] * 0.95, '预测起点',
             rotation=90, va='top', ha='right', fontweight='bold')

    plt.title(f'自定义时间段回测: {TEST_START} 至 {TEST_END} (MAPE={mape:.2f}%)',
              fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('株高 (cm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存
    os.makedirs('output', exist_ok=True)
    safe_name = f"custom_test_{TEST_START.replace('-', '')}_{TEST_END.replace('-', '')}"
    plt.savefig(f'output/{safe_name}.png', dpi=200)
    print(f"✅ 图表已保存至: output/{safe_name}.png")

    # 11. 打印详细结果
    print("\n📅 详细预测对比:")
    print("-" * 55)
    print(f"{'日期':<12} {'真实值(cm)':<12} {'预测值(cm)':<12} {'绝对误差':<10} {'相对误差(%)':<12}")
    print("-" * 55)
    for i, date in enumerate(true_dates):
        true_val = true_values[i]
        pred_val = pred_values[i]
        abs_err = abs(true_val - pred_val)
        rel_err = (abs_err / true_val) * 100
        print(f"{date.strftime('%m-%d'):<12} {true_val:<12.2f} {pred_val:<12.2f} {abs_err:<10.2f} {rel_err:<12.2f}")

    plt.show()


if __name__ == '__main__':
    main()