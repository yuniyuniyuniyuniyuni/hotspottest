import io
import re
import json
import joblib
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBRegressor

# ===== 설정 =====
MODEL_PATH = "model.joblib"
FEATURE_PATH = "feature_columns.json"

# 학습/추론 데이터 경로 분리
from pathlib import Path

# Path 객체 생성 시 슬래시를 사용해도 내부적으로 OS에 맞게 처리됩니다.
DATA_PATH_2022 = Path("..") / "data" / "predict_db" / "행정동X업종_통합_20221.csv"
DATA_PATH_2025 = Path("..") / "data" / "predict_db" / "행정동X업종_통합_20251.csv" # 2025년 추론 데이터 파일명 (필요 시 수정)

SEED = 42

# ===== 앱 생성 & CORS =====
app = FastAPI(title="HotSpot Survival API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 전역 상태 =====
model: Optional[XGBRegressor] = None
feature_columns: List[str] = []
obj_ref_cols: List[str] = []

# 원본 데이터 (학습용과 추론용 분리)
raw_df_2022: Optional[pd.DataFrame] = None
raw_df_2025: Optional[pd.DataFrame] = None

# ===== 유틸 =====
def drop_identifier_cols(df: pd.DataFrame) -> pd.DataFrame:
    """코드/이름 라벨성 컬럼 제거."""
    drop_cols = [c for c in df.columns if ("코드" in c) or c.endswith("_명")]
    return df.drop(columns=drop_cols, errors="ignore")

def build_X_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임 기반으로 학습 시 사용한 X 데이터프레임을 구성."""
    df = drop_identifier_cols(df)

    if "기준_년분기_코드" in df.columns:
        s = df["기준_년분기_코드"].astype(str)
        df["연도"] = s.str.extract(r"(20\\d{2})").astype(float)
        df["분기"] = s.str.extract(r"(?:20\\d{2})\\D*([1-4])").astype(float)
        df = df.drop(columns=["기준_년분기_코드"])

    # df.dropna(subset=["생존률"]) # 추론 시에는 생존률 컬럼이 없을 수 있으므로 이 로직을 삭제
    
    obj_cols = [c for c in df.select_dtypes(include="object").columns.tolist() if c in obj_ref_cols]
    df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)

    X = pd.DataFrame(columns=feature_columns)
    X = pd.concat([X, df], ignore_index=True)
    X = X.fillna(0)
    X = X[feature_columns]
    
    return X


# ===== 모델 및 데이터 로드 (서버 시작 시 한 번만 실행) =====
@app.on_event("startup")
def load_resources():
    """서버 시작 시 모델, 특성 컬럼 목록, 그리고 원본 데이터를 로드."""
    global model, feature_columns, obj_ref_cols, raw_df_2022, raw_df_2025
    
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_PATH, 'r') as f:
            data = json.load(f)
            feature_columns = data['feature_columns']
            obj_ref_cols = data['obj_ref_cols']
        
        # 2022년 데이터 로드 (모델 학습 시 사용한 스키마 확인용)
        raw_df_2022 = pd.read_csv(DATA_PATH_2022, encoding='utf-8-sig')

        # 2025년 데이터 로드 (추론 시 사용)
        raw_df_2025 = pd.read_csv(DATA_PATH_2025, encoding='utf-8-sig')

        print("Model, features, and raw data loaded successfully.")

    except Exception as e:
        print(f"Error loading resources: {e}")
        model = None
        feature_columns = []
        raw_df_2022 = None
        raw_df_2025 = None
        raise HTTPException(status_code=500, detail=f"Failed to load resources: {e}")

# ===== API 라우터 =====
class PredictSelectionPayload(BaseModel):
    dong_code: str
    industry_code: str

@app.post("/predict_by_selection")
def predict_by_selection(payload: PredictSelectionPayload):
    """
    사용자가 선택한 업종과 지역 코드를 기반으로 2025년 데이터에서 예측을 진행.
    """
    if model is None or raw_df_2025 is None:
        raise HTTPException(status_code=500, detail="Server resources not ready.")
    
    # 2025년 데이터에서 사용자가 선택한 지역과 업종 코드로 필터링
    df_filtered = raw_df_2025[
        (raw_df_2025['행정동_코드'] == int(payload.dong_code)) &
        (raw_df_2025['서비스_업종_코드'] == payload.industry_code)
    ].copy()

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No data found for the selected region and industry in the 2025 dataset.")

    try:
        X = build_X_from_df(df_filtered)
        
        # 예측
        predictions = model.predict(X)
        pred = float(predictions.mean()) # 해당 지역/업종의 평균 생존율 예측
        
        return {
            "dong_code": payload.dong_code,
            "industry_code": payload.industry_code,
            "prediction": round(pred, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# 기존 /predict_csv 엔드포인트는 그대로 유지
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if model is None or not feature_columns:
        return {"error": "Model not ready"}

    try:
        df = pd.read_csv(io.StringIO((await file.read()).decode('utf-8')))

        df = drop_identifier_cols(df)
        
        if "기준_년분기_코드" in df.columns:
            s = df["기준_년분기_코드"].astype(str)
            df["연도"] = s.str.extract(r"(20\\d{2})").astype(float)
            df["분기"] = s.str.extract(r"(?:20\\d{2})\\D*([1-4])").astype(float)
            df = df.drop(columns=["기준_년분기_코드"])

        obj_cols = [c for c in df.select_dtypes(include="object").columns.tolist() if c in obj_ref_cols]
        df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)
        
        X = pd.DataFrame(columns=feature_columns)
        X = pd.concat([X, df], ignore_index=True)
        X = X.fillna(0)
        X = X[feature_columns]

        predictions = model.predict(X)
        
        result = [
            {"row_id": i, "prediction": round(float(p), 4)}
            for i, p in enumerate(predictions)
        ]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {e}")

# (참고) 모델을 학습하고 저장하는 스크립트.
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    import json

    df = pd.read_csv(DATA_PATH_2022, encoding="utf-8-sig")
    df = df.sample(frac=0.1, random_state=SEED)

    df["생존률"] = pd.to_numeric(df["생존률"].replace({"-" : None, "": None}), errors="coerce")
    
    df = drop_identifier_cols(df)
    obj_ref_cols = df.select_dtypes(include="object").columns.tolist()
    
    if "기준_년분기_코드" in df.columns:
        s = df["기준_년분기_코드"].astype(str)
        df["연도"] = s.str.extract(r"(20\\d{2})").astype(float)
        df["분기"] = s.str.extract(r"(?:20\\d{2})\\D*([1-4])").astype(float)
        df = df.drop(columns=["기준_년분기_코드"])
    
    df = df.dropna(subset=["생존률"])
    
    df = pd.get_dummies(df, columns=obj_ref_cols, dummy_na=True)
    
    X = df.drop(columns=["생존률"])
    y = df["생존률"]
    feature_columns = X.columns.tolist()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=SEED)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    with open(FEATURE_PATH, 'w') as f:
        json.dump({"feature_columns": feature_columns, "obj_ref_cols": obj_ref_cols}, f)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"모델 학습 및 평가 완료. RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)