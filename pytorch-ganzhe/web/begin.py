# 文件名: sugarcane_dashboard.py

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
    """调用LSTM预测API (需要替换为实际的API端点和逻辑)"""
    # 示例API调用结构
    # api_url = "http://your-lstm-api/predict"
    # payload = {"features": feature_data.tolist()}
    # try:
    #     response = requests.post(api_url, json=payload)
    #     if response.status_code == 200:
    #         result = response.json()
    #         return result['predictions']
    #     else:
    #         st.error(f"API调用失败: {response.status_code}")
    #         return None
    # except requests.exceptions.RequestException as e:
    #     st.error(f"API请求异常: {e}")
    #     return None
    # --- 模拟API调用 ---
    # 模拟返回未来15天的预测数据
    import random
    last_height = feature_data[-1, -1]  # 假设最后一个特征是高度
    simulated_predictions = [last_height + random.uniform(-2, 5) for _ in range(15)]
    return simulated_predictions
    # ---


def call_yolo_detection_api(pil_image):
    """调用YOLO检测API (FastAPI服务)"""
    api_url = "http://localhost:8000/predict/"  # FastAPI服务的地址

    # 将 PIL Image 转换为字节流
    img_byte_arr = io.BytesIO()

    # 检查并转换图片模式 (RGBA -> RGB)
    if pil_image.mode in ("RGBA", "LA", "P"):
        # 创建一个白色背景的图片
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        # 将原图粘贴到背景上，丢弃透明度信息
        if pil_image.mode == "P":
            # 对于调色板模式，需要先转换
            pil_image = pil_image.convert("RGBA")
        background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ("RGBA", "LA") else None)
        pil_image_to_save = background
    else:
        # 如果不是RGBA/LA/P模式，则直接使用原图
        pil_image_to_save = pil_image

    # 保存图片到字节流 (转换为RGB后，就可以保存为JPEG)
    pil_image_to_save.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}  # 发送文件
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
    col1_pred, col2_pred = st.columns([3, 1])
    with col1_pred:
        st.info("点击下方按钮，将使用最近45天的数据调用API预测未来15天的植株高度。")
    with col2_pred:
        if st.button("执行API预测", key="run_api_prediction"):
            if st.session_state.df is not None:
                with st.spinner("正在调用LSTM预测API..."):
                    try:
                        # 准备API输入数据
                        df = st.session_state.df
                        feature_cols = [temp_col, rain_col, sun_col, soil_col, lai_col, stem_col,
                                        height_col]  # 包含高度作为序列一部分
                        last_45_days_df = df.tail(45)
                        if last_45_days_df.shape[0] < 45:
                            st.error(f"数据不足，需要至少45天的数据进行预测，当前只有 {last_45_days_df.shape[0]} 天。")
                        else:
                            X_for_api = last_45_days_df[feature_cols].values  # Shape: (45, 7)
                            # 调用API
                            api_predictions = call_lstm_prediction_api(X_for_api)
                            if api_predictions:
                                # 生成未来15天的日期
                                last_date = df['date'].iloc[-1]
                                future_dates = [last_date + timedelta(days=i) for i in range(1, 16)]

                                # 准备绘图数据
                                plot_df_hist = df[['date', height_col]].tail(45).copy()
                                plot_df_hist.rename(columns={height_col: 'plant_height'}, inplace=True)
                                plot_df_pred = pd.DataFrame(
                                    {'date': future_dates, 'plant_height': api_predictions, 'type': '预测'})
                                plot_df_hist['type'] = '历史'

                                plot_df_combined = pd.concat([plot_df_hist, plot_df_pred], ignore_index=True)

                                fig_pred = px.line(plot_df_combined, x='date', y='plant_height', color='type',
                                                   title='植株高度：历史数据与API预测对比',
                                                   labels={'plant_height': '高度 (cm)', 'type': '数据类型'})
                                fig_pred.update_layout(xaxis_title='日期', yaxis_title='植株高度 (cm)')
                                st.plotly_chart(fig_pred, use_container_width=True)

                                pred_result_df = pd.DataFrame({
                                    '预测日期': future_dates,
                                    '预测植株高度 (cm)': api_predictions
                                })
                                st.session_state.prediction_result = pred_result_df
                                st.dataframe(pred_result_df, use_container_width=True)
                                st.session_state.api_prediction_data = pred_result_df  # 存储到session_state
                                st.success("API预测完成！")
                            else:
                                st.error("API预测失败。")
                    except Exception as e:
                        st.error(f"API预测过程中出错: {e}")
            else:
                st.warning("请先加载数据。")

    if st.session_state.api_prediction_data is not None:
        st.download_button(
            label="💾 下载预测结果 CSV",
            data=st.session_state.api_prediction_data.to_csv(index=False).encode('utf-8'),
            file_name='sugarcane_height_api_prediction.csv',
            mime='text/csv',
        )

elif view_option == "YOLO病虫害检测":
    st.subheader("🪲 甘蔗病虫害YOLO检测")
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
                        # 调用API (传入 PIL Image 对象)
                        api_results = call_yolo_detection_api(st.session_state.uploaded_image)
                        if api_results:
                            st.success("API检测完成！")
                            predicted_class = api_results.get("class", "Unknown")
                            confidence = api_results.get("confidence", 0.0)

                            # 显示检测结果
                            st.subheader("检测结果:")
                            st.write(f"- **预测类别:** {predicted_class}")
                            st.write(f"- **置信度:** {confidence:.4f}")

                            st.session_state.api_detection_results = api_results  # 存储到session_state
                        else:
                            st.error("API检测失败。")
                    except Exception as e:
                        st.error(f"API检测过程中出错: {e}")
        else:
            st.warning("请先上传一张甘蔗叶片图片。")

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