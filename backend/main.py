import io
import json
import joblib
from typing import List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBRegressor


MODEL_PATH = "model.joblib"
FEATURE_PATH = "feature_columns.json"

DATA_PATH_2022 = Path("..") / "data" / "predict_db" / "행정동X업종_통합_20221.csv"
DATA_PATH_2025 = Path("..") / "data" / "predict_db" / "행정동X업종_통합_20251.csv" 

SEED = 42

model: Optional[XGBRegressor] = None
feature_columns: List[str] = []
obj_ref_cols: List[str] = []

raw_df_2022: Optional[pd.DataFrame] = None
raw_df_2025: Optional[pd.DataFrame] = None

def load_resources():
    """모델, 특성 컬럼 목록, 그리고 원본 데이터를 로드."""
    global model, feature_columns, obj_ref_cols, raw_df_2022, raw_df_2025
    
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_PATH, 'r') as f:
            data = json.load(f)
            feature_columns = data['feature_columns']
            obj_ref_cols = data['obj_ref_cols']

        raw_df_2022 = pd.read_csv(DATA_PATH_2022, encoding='utf-8-sig')

        raw_df_2025 = pd.read_csv(DATA_PATH_2025, encoding='utf-8-sig')

        print("Model, features, and raw data loaded successfully.")
        return True

    except Exception as e:
        print(f"Error loading resources: {e}")
        model = None
        feature_columns = []
        raw_df_2022 = None
        raw_df_2025 = None
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 자원 로드, 종료 시 자원 해제(필요하다면)를 관리."""
    print("Application Startup: Loading resources...")
    if not load_resources():
        print("WARNING: Resources failed to load. The server will start, but prediction endpoints will fail.")
    yield
    print("Application Shutdown: Resources cleanup complete.")


# ===== 앱 생성 & CORS (lifespan 인자 추가) =====
app = FastAPI(title="HotSpot Survival API", version="0.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def drop_identifier_cols(df: pd.DataFrame) -> pd.DataFrame:
    """코드/이름 라벨성 컬럼 제거."""
    drop_cols = [c for c in df.columns if ("코드" in c) or c.endswith("_명")]
    return df.drop(columns=drop_cols, errors="ignore")

def build_X_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임 기반으로 학습 시 사용한 X 데이터프레임을 구성."""
    df = drop_identifier_cols(df)

    if "기준_년분기_코드" in df.columns:
        s = df["기준_년분기_코드"].astype(str)
        df["연도"] = s.str.extract(r"(20\d{2})").astype(float)
        df["분기"] = s.str.extract(r"(?:20\d{2})\D*([1-4])").astype(float)
        df = df.drop(columns=["기준_년분기_코드"])
    
    obj_cols = [c for c in df.select_dtypes(include="object").columns.tolist() if c in obj_ref_cols]
    df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)

    X = pd.DataFrame(columns=feature_columns)
    X = pd.concat([X, df], ignore_index=True)
    X = X.fillna(0)
    X = X[feature_columns]
    
    return X


# ===== API 라우터 =====
@app.get("/")
def health_check():
    """서버 상태 확인 및 리소스 로드 여부 반환."""
    return {
        "status": "ok",
        "model_ready": model is not None,
        "data_2025_ready": raw_df_2025 is not None,
    }

class PredictSelectionPayload(BaseModel):
    dong_code: str
    industry_code: str

@app.post("/predict_by_selection")
def predict_by_selection(payload: PredictSelectionPayload):
    if model is None or raw_df_2025 is None:
        raise HTTPException(status_code=500, detail="Server resources not ready. Model or 2025 data failed to load.")
    

    df_filtered = raw_df_2025[
        (raw_df_2025['행정동_코드'] == int(payload.dong_code)) &
        (raw_df_2025['서비스_업종_코드'] == payload.industry_code)
    ].copy()

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No data found for the selected region and industry in the 2025 dataset.")

    try:
        X = build_X_from_df(df_filtered)
        
        predictions = model.predict(X)
        pred = float(predictions.mean())
        
        return {
            "dong_code": payload.dong_code,
            "industry_code": payload.industry_code,
            "prediction": round(pred, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if model is None or not feature_columns:
        raise HTTPException(status_code=500, detail="Model not ready")

    try:
        content = (await file.read()).decode('utf-8', errors='ignore')
        df = pd.read_csv(io.StringIO(content))

        df = drop_identifier_cols(df)
        
        if "기준_년분기_코드" in df.columns:
            s = df["기준_년분기_코드"].astype(str)
            df["연도"] = s.str.extract(r"(20\d{2})").astype(float)
            df["분기"] = s.str.extract(r"(?:20\d{2})\D*([1-4])").astype(float)
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

if __name__ == "__main__":

    
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)