# 文件名: lstm_api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from prediction_utils import predict_future_height

app = FastAPI(title="甘蔗植株高度LSTM预测API")

@app.post("/predict_height/")
async def predict_height_endpoint(data: dict):
    """
    接收包含最近45天数据的JSON，预测未来15天的植株高度。
    JSON格式应包含 'temperature', 'precipitation', 'sunshine_hours', 'soil_moisture', 'plant_height', 'leaf_area_index', 'stem_diameter' 列的数据。
    """
    try:
        # 将输入的字典转换为DataFrame
        df = pd.DataFrame(data)

        # 验证DataFrame是否包含必要的列
        required_cols = ['temperature', 'precipitation', 'sunshine_hours', 'soil_moisture', 'plant_height', 'leaf_area_index', 'stem_diameter']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(status_code=400, detail=f"输入数据缺少必要列: {required_cols}")

        # 验证数据长度
        if df.shape[0] < 45:
             raise HTTPException(status_code=400, detail=f"输入数据长度不足，需要至少45天的数据，当前只有 {df.shape[0]} 天。")

        # 调用预测函数
        result = predict_future_height(df)

        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "predictions": result['predictions']
                }
            )
        else:
            raise HTTPException(status_code=500, detail=f"预测失败: {result['error']}")

    except HTTPException:
        # 重新抛出 FastAPI 的 HTTPException
        raise
    except Exception as e:
        # 处理其他所有异常
        print(f"预测过程中出错: {e}")
        raise HTTPException(status_code=500, detail=f"预测过程中出错: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "甘蔗植株高度LSTM预测API服务运行中"}

if __name__ == "__main__":
    import uvicorn
    # 启动服务，监听 8002 端口
    uvicorn.run(app, host="0.0.0.0", port=8002)