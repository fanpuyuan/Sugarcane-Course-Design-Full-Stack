import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置随机种子以确保结果可复现
np.random.seed(42)


def generate_sugar_crop_time_series(start_date='2023-03-01', days=180):
    """
    生成糖料作物（甘蔗）生长时序数据（模拟）

    参数:
        start_date (str): 起始日期，格式 'YYYY-MM-DD'
        days (int): 生成天数，默认180天（约一个生长季）

    返回:
        pd.DataFrame: 包含日期和多维特征的时序数据框
    """
    # 生成日期范围
    dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(days)]

    # 模拟温度（°C）：春季升温，夏季高温，有日波动
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    base_temp = 10 + 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # 年周期变化
    temp_noise = np.random.normal(0, 2, days)  # 日波动
    temperature = base_temp + temp_noise
    temperature = np.clip(temperature, 5, 40)  # 限制合理范围

    # 模拟降水（mm）：随机事件，大部分为0，偶尔有降雨
    precipitation = np.zeros(days)
    rain_days = np.random.choice(days, size=int(days * 0.25), replace=False)  # 25%天数有雨
    precipitation[rain_days] = np.random.exponential(scale=8.0, size=len(rain_days))  # 指数分布模拟降雨量
    precipitation = np.round(precipitation, 1)

    # 模拟日照时数（小时）：随季节变化，夏季长，冬季短
    sunshine_base = 6 + 3 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
    sunshine_noise = np.random.normal(0, 0.8, days)
    sunshine_hours = sunshine_base + sunshine_noise
    sunshine_hours = np.clip(sunshine_hours, 2, 12)
    sunshine_hours = np.round(sunshine_hours, 1)

    # 模拟土壤湿度（%）：受降水和蒸发影响
    soil_moisture = np.zeros(days)
    soil_moisture[0] = 25.0  # 初始值
    for i in range(1, days):
        evaporation = 0.1 * temperature[i] + 0.05 * sunshine_hours[i]
        change = precipitation[i] * 0.3 - evaporation * 0.2
        soil_moisture[i] = np.clip(soil_moisture[i - 1] + change, 5, 50)
    soil_moisture = np.round(soil_moisture, 1)

    # =============== 修正：株高（plant_height）===============
    # 使用逻辑斯蒂曲线定义理论生长（最大300 cm）
    t = np.linspace(0, 1, days)
    theoretical_height = 300 / (1 + np.exp(-8 * (t - 0.5)))  # S型曲线，拐点在中期

    # 计算每日理论生长量（必须 ≥0）
    daily_growth_theoretical = np.diff(theoretical_height, prepend=theoretical_height[0])

    # 在每日生长量上添加噪声（允许±2 cm波动，但不能为负）
    daily_growth_noisy = daily_growth_theoretical + np.random.normal(0, 0.8, days)
    daily_growth_noisy = np.clip(daily_growth_noisy, 0, None)  # 确保非负

    # 累加得到实际株高（严格非递减）
    plant_height = np.cumsum(daily_growth_noisy)
    plant_height = np.round(plant_height, 2)

    # =============== 修正：叶面积指数（LAI）===============
    # 使用饱和函数：LAI 随株高增长但趋于稳定（甘蔗 LAI 通常 4–7）
    lai = 6.5 * (1 - np.exp(-0.015 * plant_height))  # 渐近线 LAI ≈ 6.5
    lai += np.random.normal(0, 0.15, days)  # 小幅测量/个体差异噪声
    lai = np.clip(lai, 0.3, 8.0)
    lai = np.round(lai, 2)

    # =============== 修正：茎粗（stem_diameter）===============
    # 从约 20 mm 缓慢增长到 35 mm
    stem_diameter = 20.0 + 15.0 * (t ** 0.7)
    stem_diameter += np.random.normal(0, 0.4, days)  # 小噪声
    stem_diameter = np.clip(stem_diameter, 18, 40)
    stem_diameter = np.round(stem_diameter, 2)

    # 构建DataFrame
    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'temperature': np.round(temperature, 2),
        'precipitation': precipitation,
        'sunshine_hours': sunshine_hours,
        'soil_moisture': soil_moisture,
        'plant_height': plant_height,
        'leaf_area_index': lai,
        'stem_diameter': stem_diameter
    })

    # 随机引入约5%的缺失值（模拟传感器故障或人工观测遗漏）
    total_cells = df.shape[0] * (df.shape[1] - 1)  # 不包括date列
    num_missing = int(total_cells * 0.05)
    missing_indices = np.random.choice(df.index, size=num_missing, replace=True)
    missing_cols = np.random.choice(df.columns[1:], size=num_missing, replace=True)

    for idx, col in zip(missing_indices, missing_cols):
        df.loc[idx, col] = np.nan

    return df


# 使用示例
if __name__ == "__main__":
    # 生成180天的甘蔗生长数据
    simulated_data = generate_sugar_crop_time_series(start_date='2023-03-01', days=180)

    # 保存为CSV文件
    simulated_data.to_csv('sugar_cane_growth_data.csv', index=False)
    print("✅ 模拟数据已生成并保存为 'sugar_cane_growth_data.csv'")
    print(f"数据形状: {simulated_data.shape}")
    print(f"缺失值数量: {simulated_data.isnull().sum().sum()}")
    print("\n前10行数据预览:")
    print(simulated_data.head(10))

    # 验证株高是否非递减（应全为 True）
    height_non_decreasing = (simulated_data['plant_height'].dropna().diff().fillna(0) >= -1e-6).all()
    print(f"\n株高序列是否非递减？{'是' if height_non_decreasing else '否'}")