# 文件名: sugarcane_dashboard.py
import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pyecharts import options as opts
from pyecharts.charts import Map
import streamlit.components.v1 as components
import requests  # 用于API调用
from PIL import Image  # 用于处理上传的图片
import io  # 用于转换图片为字节流

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="甘蔗农事数据大屏",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🌱 甘蔗农事数据大屏")
st.markdown("---")

# --- 2. 初始化会话状态 ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'video_uploaded' not in st.session_state:
    st.session_state.video_uploaded = False
if 'uploaded_video' not in st.session_state:
    st.session_state.uploaded_video = None
if 'api_prediction_data' not in st.session_state:
    st.session_state.api_prediction_data = None
if 'api_detection_results' not in st.session_state:
    st.session_state.api_detection_results = None


# --- 3. 列名映射辅助函数 ---
def find_column_name(df, possible_names):
    """根据可能的列名列表，在DataFrame中查找实际存在的列名"""
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"无法在数据中找到列名之一: {possible_names}")


# --- 4. 预加载数据 ---
@st.cache_data
def load_preloaded_csv(file_path):
    """加载预设的CSV数据"""
    try:
        df = pd.read_csv(file_path)
        date_col = find_column_name(df, ['date', 'Date', 'datetime', 'DateTime', 'time', 'Time'])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        df.rename(columns={date_col: 'date'}, inplace=True)
        return df
    except FileNotFoundError:
        st.warning(f"预设数据文件 {file_path} 未找到。请稍后手动上传。")
        return None
    except Exception as e:
        st.error(f"加载预设CSV数据时出错: {e}")
        return None


# --- 5. 手动上传数据函数 ---
def load_csv_data(file_path_or_buffer):
    """加载上传的CSV数据 (接受 Streamlit 上传的文件对象)"""
    try:
        df = pd.read_csv(file_path_or_buffer)
        date_col = find_column_name(df, ['date', 'Date', 'datetime', 'DateTime', 'time', 'Time'])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        df.rename(columns={date_col: 'date'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"加载上传的CSV数据时出错: {e}")
        return None


# --- 6. API 调用辅助函数 (示例) ---
def call_lstm_prediction_api(feature_data):
    """调用LSTM预测API (FastAPI服务)"""
    api_url = "http://localhost:8002/predict_height/" # LSTM API服务的地址

    # feature_data 是 numpy array，形状为 (45, 7)
    # 需要将其转换为 FastAPI 期望的 JSON 格式
    # 假设列顺序为 ['temperature', 'precipitation', 'sunshine_hours', 'soil_moisture', 'plant_height', 'leaf_area_index', 'stem_diameter']
    # 你需要根据你的实际数据列顺序调整
    feature_df = pd.DataFrame(feature_data, columns=['temperature', 'precipitation', 'sunshine_hours', 'soil_moisture', 'plant_height', 'leaf_area_index', 'stem_diameter'])
    json_payload = feature_df.to_dict(orient='list') # 转换为列名: [values] 的字典

    try:
        response = requests.post(api_url, json=json_payload) # 发送 JSON 数据
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("predictions") # 返回预测列表
            else:
                st.error(f"API返回错误: {result.get('error', '未知错误')}")
                return None
        else:
            st.error(f"API调用失败: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API请求异常: {e}")
        return None


def call_yolo_detection_api(pil_image):
    """调用YOLO检测API (FastAPI服务) - 图片"""
    api_url = "http://localhost:8000/predict_image/"  # 注意端点变化

    # 将 PIL Image 转换为字节流
    img_byte_arr = io.BytesIO()

    # 检查并转换图片模式 (RGBA -> RGB)
    if pil_image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        if pil_image.mode == "P":
            pil_image = pil_image.convert("RGBA")
        background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ("RGBA", "LA") else None)
        pil_image_to_save = background
    else:
        pil_image_to_save = pil_image

    pil_image_to_save.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
    try:
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("prediction")
            else:
                st.error(f"API返回错误: {result.get('error', '未知错误')}")
                return None
        else:
            st.error(f"API调用失败: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API请求异常: {e}")
        return None


def call_yolo_video_detection_api(uploaded_video_file):
    """调用YOLO检测API (FastAPI服务) - 视频"""
    api_url = "http://localhost:8000/predict_video/"  # 注意端点变化

    # 上传文件对象可以直接使用
    files = {'file': uploaded_video_file}
    try:
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            # 直接返回视频内容
            return response.content
        else:
            st.error(f"API调用失败: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API请求异常: {e}")
        return None


# --- 7. 加载预设数据 ---
st.session_state.df = load_preloaded_csv("sugar_crop_data.csv")

# --- 8. 侧边栏功能选择器 ---
st.sidebar.header("📋 选择功能")
view_option = st.sidebar.radio("选择要查看的内容", ("数据可视化", "YOLO病虫害检测"))

# --- 9. 主界面内容 ---
# --- 9.1 数据概览 ---
if st.session_state.df is not None:
    df = st.session_state.df
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 农事数据可视化")
    with col2:
        st.subheader("数据概览")
        st.metric(label="数据点总数", value=df.shape[0])
        st.metric(label="时间跨度", value=f"{df['date'].min().date()} 至 {df['date'].max().date()}")
        st.info(f"识别到的列: {list(df.columns)}")

    # 自动识别各列名称
    try:
        temp_col = find_column_name(df, ['temperature', 'temp', 'Temperature'])
        rain_col = find_column_name(df, ['precipitation', 'rainfall', 'rain', 'Precipitation'])
        sun_col = find_column_name(df, ['sunshine_hours', 'sunlight_hours', 'sun_hours', 'Sunshine_hours'])
        soil_col = find_column_name(df, ['soil_moisture', 'Soil_moisture'])
        height_col = find_column_name(df, ['plant_height', 'height', 'Plant_height'])
        lai_col = find_column_name(df, ['leaf_area_index', 'Leaf_area_index'])
        stem_col = find_column_name(df, ['stem_diameter', 'Stem_diameter'])
    except ValueError as e:
        st.error(f"CSV文件列名识别失败: {e}")
        st.stop()

    # 创建多个图表
    fig_temp = px.line(df, x='date', y=temp_col, title='温度变化趋势', markers=True)
    fig_rain = px.bar(df, x='date', y=rain_col, title='降水量柱状图')
    fig_sun = px.line(df, x='date', y=sun_col, title='日照时数变化', line_shape='spline')
    fig_soil = px.line(df, x='date', y=soil_col, title='土壤湿度变化趋势', markers=True)
    fig_height = px.line(df, x='date', y=height_col, title='植株高度变化趋势', markers=True, line_shape='spline')
    fig_lai = px.line(df, x='date', y=lai_col, title='叶面积指数变化', markers=True)
    fig_stem = px.line(df, x='date', y=stem_col, title='茎粗变化趋势', markers=True)

    col1_viz, col2_viz = st.columns(2)
    with col1_viz:
        st.plotly_chart(fig_temp, use_container_width=True)
        st.plotly_chart(fig_sun, use_container_width=True)
        st.plotly_chart(fig_height, use_container_width=True)
        st.plotly_chart(fig_stem, use_container_width=True)
    with col2_viz:
        st.plotly_chart(fig_rain, use_container_width=True)
        st.plotly_chart(fig_soil, use_container_width=True)
        st.plotly_chart(fig_lai, use_container_width=True)

    st.markdown("---")

    # --- 9.2 广西甘蔗产量地图 ---
    st.subheader("🗺️ 广西甘蔗产量地图")
    # 甘蔗产量数据（标准地市名称）
    map_data = [
        ("崇左市", 13500000),
        ("南宁市", 9000000),
        ("来宾市", 8000000),
        ("柳州市", 3500000),
        ("百色市", 2500000),
        ("贵港市", 1800000),
        ("钦州市", 1200000),
        ("河池市", 800000),
        ("玉林市", 400000),
    ]

    # 创建地图
    map_chart = (
        Map()
        .add("甘蔗产量（吨）", map_data, "广西")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="广西甘蔗产量分布"),
            visualmap_opts=opts.VisualMapOpts(
                min_=0,
                max_=14000000,
                is_piecewise=True,
                pieces=[
                    {"min": 10000000, "label": "≥1000万吨", "color": "#7F1406"},
                    {"min": 5000000, "max": 9999999, "label": "500–999万吨", "color": "#CB4B2A"},
                    {"min": 1000000, "max": 4999999, "label": "100–499万吨", "color": "#F29F72"},
                    {"max": 999999, "label": "<100万吨", "color": "#FCE5D2"},
                ],
            ),
        )
    )

    components.html(
        map_chart.render_embed(),
        height=600,
        scrolling=False
    )

    st.markdown("---")

# --- 9.3 根据侧边栏选择显示内容 ---
if view_option == "数据可视化":
    st.subheader("📈 LSTM 植株高度预测")
    st.info("点击下方按钮，将使用最近45天的数据调用API预测未来15天的植株高度。")

    if st.button("执行API预测", key="run_api_prediction"):
        if st.session_state.df is not None:
            with st.spinner("正在调用LSTM预测API..."):
                try:
                    # --- 调试：打印数据信息 ---
                    df = st.session_state.df
                    print("\n--- DEBUG: Full DataFrame Info ---")
                    print(f"DataFrame shape: {df.shape}")
                    print(f"DataFrame columns: {list(df.columns)}")
                    print(f"DataFrame head:\n{df.head()}")
                    print(f"DataFrame tail:\n{df.tail()}")
                    print(f"DataFrame dtypes:\n{df.dtypes}")
                    print(f"DataFrame describe:\n{df.describe()}")
                    print(f"DataFrame isnull().sum():\n{df.isnull().sum()}")
                    # --- 调试结束 ---

                    # 准备API输入数据
                    # 自动识别各列名称 (确保这部分代码与数据可视化部分一致)
                    temp_col = find_column_name(df, ['temperature', 'temp', 'Temperature'])
                    rain_col = find_column_name(df, ['precipitation', 'rainfall', 'rain', 'Precipitation'])
                    sun_col = find_column_name(df, ['sunshine_hours', 'sunlight_hours', 'sun_hours', 'Sunshine_hours'])
                    soil_col = find_column_name(df, ['soil_moisture', 'Soil_moisture'])
                    height_col = find_column_name(df, ['plant_height', 'height', 'Plant_height'])
                    lai_col = find_column_name(df, ['leaf_area_index', 'Leaf_area_index'])
                    stem_col = find_column_name(df, ['stem_diameter', 'Stem_diameter'])

                    feature_cols = [temp_col, rain_col, sun_col, soil_col, height_col, lai_col, stem_col] # 7个特征 (包含高度)
                    last_45_days_df = df.tail(45)

                    # --- 数据填充：处理 last_45_days_df 中的 nan 值 ---
                    # 这里使用简单的前向填充和后向填充作为示例
                    # 你可以根据你的 fill_missing_agri 逻辑进行更复杂的填充
                    last_45_days_df_filled = last_45_days_df.copy()
                    last_45_days_df_filled = last_45_days_df_filled.ffill().bfill()
                    # 或者使用均值填充
                    # last_45_days_df_filled = last_45_days_df_filled.fillna(last_45_days_df_filled.mean())
                    # 或者使用中位数填充
                    # last_45_days_df_filled = last_45_days_df_filled.fillna(last_45_days_df_filled.median())

                    # --- 调试：打印填充后的最后45天数据 ---
                    print("\n--- DEBUG: Last 45 Days DataFrame (After Fill) Info ---")
                    print(f"Filled DataFrame shape: {last_45_days_df_filled.shape}")
                    print(f"Filled DataFrame head:\n{last_45_days_df_filled.head()}")
                    print(f"Filled DataFrame tail:\n{last_45_days_df_filled.tail()}")
                    print(f"Filled DataFrame isnull().sum():\n{last_45_days_df_filled.isnull().sum()}")
                    # --- 调试结束 ---

                    if last_45_days_df_filled.shape[0] < 45:
                        st.error(f"数据不足，需要至少45天的数据进行预测，当前只有 {last_45_days_df_filled.shape[0]} 天。")
                    else:
                        X_for_api = last_45_days_df_filled[feature_cols].values # Shape: (45, 7)

                        # --- 调试：打印转换为 numpy array 后的数据 ---
                        print("\n--- DEBUG: Numpy Array (X_for_api) Info ---")
                        print(f"X_for_api shape: {X_for_api.shape}")
                        print(f"X_for_api dtype: {X_for_api.dtype}")
                        print(f"X_for_api head (first 5 rows):\n{X_for_api[:5]}")
                        print(f"X_for_api tail (last 5 rows):\n{X_for_api[-5:]}")
                        print(f"X_for_api contains inf: {np.any(np.isinf(X_for_api))}")
                        print(f"X_for_api contains nan: {np.any(np.isnan(X_for_api))}")
                        print(f"X_for_api min/max: {np.min(X_for_api)}, {np.max(X_for_api)}")
                        # --- 调试结束 ---

                        # --- 检查并处理 inf 和 nan 值 (在发送到API前再次确认) ---
                        if np.any(np.isinf(X_for_api)) or np.any(np.isnan(X_for_api)):
                            st.error("处理后的输入数据仍包含无穷大 (inf) 或非数字 (nan) 值，无法进行预测。请检查数据填充逻辑。")
                            print("DEBUG: feature_data contains inf or nan after fill") # 可选：打印到控制台调试
                            print(X_for_api) # 可选：打印具体数据
                            # 不使用 return None 或 break，直接跳过后续的API调用
                        else:
                            # 调用API
                            api_predictions = call_lstm_prediction_api(X_for_api)
                            if api_predictions:
                                # --- 修改可视化逻辑 ---
                                if df.shape[0] >= 60: # 确保有足够的数据
                                    # 获取用于预测的历史数据段 (45天)
                                    historical_data_for_pred = df.iloc[-60:-15]
                                    # 获取对应的真实未来数据段 (15天)
                                    true_future_data = df.iloc[-15:]

                                    # 准备API输入数据 (从历史数据段提取特征)
                                    X_for_pred_check = historical_data_for_pred[feature_cols].values
                                    # 再次填充 (以防万一)
                                    X_for_pred_check_df = pd.DataFrame(X_for_pred_check, columns=feature_cols)
                                    X_for_pred_check_df = X_for_pred_check_df.ffill().bfill()
                                    X_for_pred_check_filled = X_for_pred_check_df.values

                                    # 调用API获取预测值 (用于对比)
                                    historical_api_predictions = call_lstm_prediction_api(X_for_pred_check_filled)

                                    if historical_api_predictions:
                                        # 准备绘图数据
                                        plot_df_true = true_future_data[['date', height_col]].copy()
                                        plot_df_true.rename(columns={height_col: 'plant_height'}, inplace=True)
                                        plot_df_true['type'] = '真实值'

                                        future_dates_for_check = true_future_data['date'].tolist()
                                        plot_df_pred_check = pd.DataFrame({
                                            'date': future_dates_for_check,
                                            'plant_height': historical_api_predictions,
                                            'type': '预测值'
                                        })

                                        plot_df_comparison = pd.concat([plot_df_true, plot_df_pred_check], ignore_index=True)

                                        # 创建对比图 (全宽)
                                        fig_comparison = px.line(
                                            plot_df_comparison,
                                            x='date',
                                            y='plant_height',
                                            color='type',
                                            title='植株高度：预测值 vs 真实值 (历史回测)',
                                            labels={'plant_height': '高度 (cm)', 'type': '数据类型'}
                                        )
                                        fig_comparison.update_layout(xaxis_title='日期', yaxis_title='植株高度 (cm)')
                                        st.plotly_chart(fig_comparison, use_container_width=True) # 确保全宽

                                        # 显示预测结果表格 (用于回测)
                                        pred_check_result_df = pd.DataFrame({
                                            '日期': future_dates_for_check,
                                            '真实植株高度 (cm)': plot_df_true['plant_height'].tolist(),
                                            '预测植株高度 (cm)': historical_api_predictions
                                        })
                                        st.dataframe(pred_check_result_df, use_container_width=True)

                                        # 计算 MAPE
                                        true_values = plot_df_true['plant_height'].values
                                        pred_values = np.array(historical_api_predictions)
                                        mape = np.mean(np.abs((true_values - pred_values) / np.clip(np.abs(true_values), 1e-6, None))) * 100
                                        st.info(f"历史回测 MAPE: {mape:.2f}%")

                                        # 生成未来15天的日期
                                        last_date = df['date'].iloc[-1]
                                        future_dates = [last_date + timedelta(days=i) for i in range(1, 16)]

                                        # --- 修改未来预测图的数据准备：只包含预测值 ---
                                        # 准备绘图数据 (仅预测)
                                        plot_df_pred_only = pd.DataFrame({
                                            'date': future_dates,
                                            'plant_height': api_predictions
                                        })

                                        # 创建未来预测图 (只显示预测值，全宽)
                                        fig_pred = px.line(
                                            plot_df_pred_only,
                                            x='date',
                                            y='plant_height',
                                            title='植株高度：未来15天预测',
                                            labels={'plant_height': '高度 (cm)', 'date': '日期'}
                                        )
                                        fig_pred.update_layout(xaxis_title='日期', yaxis_title='植株高度 (cm)')
                                        st.plotly_chart(fig_pred, use_container_width=True)  # 确保全宽

                                        # --- 修改第二张表格：只展示预测值 ---
                                        pred_result_df = pd.DataFrame({
                                            '预测日期': future_dates,
                                            '预测植株高度 (cm)': api_predictions
                                        })
                                        st.dataframe(pred_result_df, use_container_width=True)
                                        # --- 表格修改结束 ---

                                        st.session_state.api_prediction_data = pred_result_df  # 存储到session_state
                                        st.success("API预测完成！")
                                    else:
                                        st.error("API预测失败。")
                except Exception as e:
                    st.error(f"API预测过程中出错: {e}")
        else:
            st.warning("请先加载数据。")

# --- 10. 页面底部：数据导入和页面切换 ---
    # with col2_pred:
    #     if st.button("执行API预测", key="run_api_prediction"):
    #         if st.session_state.df is not None:
    #             with st.spinner("正在调用LSTM预测API..."):
    #                 try:
    #                     # 准备API输入数据
    #                     df = st.session_state.df
    #                     feature_cols = [temp_col, rain_col, sun_col, soil_col, lai_col, stem_col,
    #                                     height_col]  # 包含高度作为序列一部分
    #                     last_45_days_df = df.tail(45)
    #                     if last_45_days_df.shape[0] < 45:
    #                         st.error(f"数据不足，需要至少45天的数据进行预测，当前只有 {last_45_days_df.shape[0]} 天。")
    #                     else:
    #                         X_for_api = last_45_days_df[feature_cols].values  # Shape: (45, 7)
    #                         # 调用API
    #                         api_predictions = call_lstm_prediction_api(X_for_api)
    #                         if api_predictions:
    #                             # 生成未来15天的日期
    #                             last_date = df['date'].iloc[-1]
    #                             future_dates = [last_date + timedelta(days=i) for i in range(1, 16)]
    #
    #                             # 准备绘图数据
    #                             plot_df_hist = df[['date', height_col]].tail(45).copy()
    #                             plot_df_hist.rename(columns={height_col: 'plant_height'}, inplace=True)
    #                             plot_df_pred = pd.DataFrame(
    #                                 {'date': future_dates, 'plant_height': api_predictions, 'type': '预测'})
    #                             plot_df_hist['type'] = '历史'
    #
    #                             plot_df_combined = pd.concat([plot_df_hist, plot_df_pred], ignore_index=True)
    #
    #                             fig_pred = px.line(plot_df_combined, x='date', y='plant_height', color='type',
    #                                                title='植株高度：历史数据与API预测对比',
    #                                                labels={'plant_height': '高度 (cm)', 'type': '数据类型'})
    #                             fig_pred.update_layout(xaxis_title='日期', yaxis_title='植株高度 (cm)')
    #                             st.plotly_chart(fig_pred, use_container_width=True)
    #
    #                             pred_result_df = pd.DataFrame({
    #                                 '预测日期': future_dates,
    #                                 '预测植株高度 (cm)': api_predictions
    #                             })
    #                             st.session_state.prediction_result = pred_result_df
    #                             st.dataframe(pred_result_df, use_container_width=True)
    #                             st.session_state.api_prediction_data = pred_result_df  # 存储到session_state
    #                             st.success("API预测完成！")
    #                         else:
    #                             st.error("API预测失败。")
    #                 except Exception as e:
    #                     st.error(f"API预测过程中出错: {e}")
    #         else:
    #             st.warning("请先加载数据。")

    if st.session_state.api_prediction_data is not None:
        st.download_button(
            label="💾 下载预测结果 CSV",
            data=st.session_state.api_prediction_data.to_csv(index=False).encode('utf-8'),
            file_name='sugarcane_height_api_prediction.csv',
            mime='text/csv',
        )

elif view_option == "YOLO病虫害检测":
    st.subheader("🪲 甘蔗病虫害YOLO检测")

    detection_type = st.radio("选择检测类型", ("图片检测 (API)", "视频检测 (API)"), key="detection_type_radio")

    if detection_type == "图片检测 (API)":
        col1_det, col2_det = st.columns(2)
        with col1_det:
            uploaded_image = st.file_uploader("上传甘蔗叶片图片进行病虫害检测", type=["jpg", "jpeg", "png"],
                                              key="image_uploader")
            if uploaded_image is not None:
                st.session_state.uploaded_image = Image.open(uploaded_image)
                st.image(st.session_state.uploaded_image, caption='上传的图片', use_column_width=True)
                st.session_state.image_uploaded = True

        with col2_det:
            if st.session_state.image_uploaded:
                if st.button("执行API检测", key="run_api_detection"):
                    with st.spinner("正在调用YOLO检测API..."):
                        try:
                            api_results = call_yolo_detection_api(st.session_state.uploaded_image)
                            if api_results:
                                st.success("API检测完成！")
                                predicted_class = api_results.get("class", "Unknown")
                                confidence = api_results.get("confidence", 0.0)

                                st.subheader("检测结果:")
                                st.write(f"- **预测类别:** {predicted_class}")
                                st.write(f"- **置信度:** {confidence:.4f}")

                                st.session_state.api_detection_results = api_results
                            else:
                                st.error("API检测失败。")
                        except Exception as e:
                            st.error(f"API检测过程中出错: {e}")
            else:
                st.warning("请先上传一张甘蔗叶片图片。")


    elif detection_type == "视频检测 (API)":

        st.info("上传视频文件或输入RTSP流地址进行检测。")

        video_source_type = st.selectbox("选择视频源类型", ("上传视频文件", "RTSP流地址"), key="video_source_type_api")

        video_file = None

        rtsp_url = ""

        if video_source_type == "上传视频文件":

            video_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"], key="video_uploader_api")

        else:  # RTSP流地址

            rtsp_url = st.text_input("输入RTSP流地址", key="rtsp_url_input")

        if st.button("开始API检测", key="start_api_video_detection"):

            if (video_source_type == "上传视频文件" and not video_file) or \
 \
                    (video_source_type == "RTSP流地址" and not rtsp_url):
                st.error("请提供视频源！")

        else:

            api_url = "http://localhost:8001/process_video/"

            files = {'file': video_file} if video_file else None

            params = {'rtsp_url': rtsp_url} if rtsp_url else {}

            with st.spinner("正在启动视频检测任务..."):

                try:

                    import requests

                    response = requests.post(api_url, files=files, params=params)

                    if response.status_code == 200:

                        task_info = response.json()

                        task_id = task_info['task_id']

                        st.success(f"任务已启动！任务ID: {task_id}")

                        # 轮询状态

                        status_placeholder = st.empty()

                        results_placeholder = st.empty()

                        progress_bar = st.progress(0)

                        video_placeholder = st.empty()  # 新增：用于显示视频

                        while True:

                            status_response = requests.get(f"http://localhost:8001/task_status/{task_id}")

                            if status_response.status_code == 200:

                                status_data = status_response.json()

                                status = status_data['status']

                                progress = status_data['progress']

                                results = status_data['results']

                                error = status_data['error']

                                output_video_filename = status_data.get('output_video_filename')  # 获取文件名

                                progress_bar.progress(progress)

                                status_placeholder.info(f"任务状态: {status}, 进度: {progress}%")

                                if status == 'completed':

                                    results_placeholder.success("检测完成！")

                                    results_placeholder.json(results)  # 显示结果

                                    # 检查是否有视频文件名返回

                                    if output_video_filename:

                                        # 构建视频播放URL

                                        video_url = f"http://localhost:8001/static/{output_video_filename}"

                                        # 在占位符中播放视频

                                        video_placeholder.video(video_url)

                                    else:

                                        video_placeholder.warning("处理后的视频文件名未返回。")

                                    break

                                elif status == 'failed':

                                    results_placeholder.error(f"任务失败: {error}")

                                    break

                                elif status == 'processing':

                                    if results:
                                        latest_results = results[-5:]

                                        results_placeholder.json(latest_results)

                                    time.sleep(2)

                                else:

                                    time.sleep(2)

                            else:

                                st.error(f"获取任务状态失败: {status_response.status_code}")

                                break

                    else:

                        st.error(f"启动任务失败: {response.status_code} - {response.text}")

                except Exception as e:

                    st.error(f"启动API检测过程中出错: {e}")

# --- 10. 页面底部：数据导入和页面切换 ---

# --- 10. 页面底部：数据导入和页面切换 ---
st.markdown("---")
st.subheader("🔧 数据管理与页面切换")

# 数据导入
uploaded_file = st.file_uploader("上传包含甘蔗农事数据的CSV文件 (将覆盖预加载数据)", type=["csv"],
                                 key="csv_uploader_bottom")
if uploaded_file is not None:
    st.session_state.df = load_csv_data(uploaded_file)  # 使用修正后的函数
    if st.session_state.df is not None:
        st.success("数据上传并加载成功！")

# 页面切换
st.info("当前显示的是甘蔗农事数据大屏。如果需要切换到其他功能页面，请在此处选择或重新运行应用。")

# --- 11. 侧边栏信息 ---
st.sidebar.markdown("---")
st.sidebar.text("Powered by Streamlit, PyEcharts, API")