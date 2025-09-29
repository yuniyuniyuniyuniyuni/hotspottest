import pandas as pd
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# --- 전역 변수 ---
# 배치 예측 결과가 저장될 DataFrame
predictions_db: Optional[pd.DataFrame] = None

# 예측 결과 파일 경로 (실제 경로에 맞게 수정해주세요)
PREDICTIONS_PATH = Path("..") / "data" / "predict_db" / "predictions_2025.csv"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버가 시작될 때 미리 계산된 예측 결과 파일을 로드합니다."""
    global predictions_db
    try:
        predictions_db = pd.read_csv(PREDICTIONS_PATH)
        print(f"✅ 성공: '{PREDICTIONS_PATH}'에서 예측 결과를 로드했습니다.")
        print(f"로드된 데이터 수: {len(predictions_db)}개")
    except FileNotFoundError:
        print(f"❌ 오류: 예측 결과 파일('{PREDICTIONS_PATH}')을 찾을 수 없습니다.")
        predictions_db = None
    except Exception as e:
        print(f"❌ 오류: 예측 결과 로드 중 예외 발생: {e}")
        predictions_db = None
    yield
    print("Application Shutdown.")


# ===== 앱 생성 & CORS =====
app = FastAPI(
    title="HotSpot Survival API",
    version="0.2", # 버전 업데이트
    description="미리 계산된 상권 생존율 예측 데이터를 제공합니다.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== API 라우터 =====
@app.get("/")
def health_check():
    """서버 상태와 예측 데이터 로드 여부를 확인합니다."""
    return {
        "status": "ok",
        "prediction_data_ready": predictions_db is not None,
    }

class PredictSelectionPayload(BaseModel):
    dong_code: str
    industry_code: str

@app.post("/predict_by_selection")
def predict_by_selection(payload: PredictSelectionPayload):
    """미리 계산된 데이터베이스에서 행정동/업종 코드에 맞는 예측 결과를 조회합니다."""
    if predictions_db is None:
        raise HTTPException(status_code=503, detail="서버 리소스(예측 DB)가 준비되지 않았습니다. 서버 로그를 확인해주세요.")

    try:
        # DataFrame에서 조건에 맞는 결과 조회
        result_row = predictions_db[
            (predictions_db['행정동_코드'] == int(payload.dong_code)) &
            (predictions_db['서비스_업종_코드'] == payload.industry_code)
        ]

        if result_row.empty:
            raise HTTPException(status_code=404, detail="선택한 지역과 업종에 대한 데이터를 찾을 수 없습니다.")

        # 조회된 첫 번째 행의 데이터를 사용
        prediction_data = result_row.iloc[0]

        # '점포당_매출_금액_예측' 컬럼이 있는지 확인하고 값을 가져옴
        if '점포당_매출_금액_예측' not in prediction_data:
             raise HTTPException(status_code=500, detail="결과 파일에 '점포당_매출_금액_예측' 컬럼이 없습니다.")

        prediction_value = prediction_data['점포당_매출_금액_예측']

        # 결과 반환 (MWS 점수 등 다른 컬럼이 있다면 여기서 추가 가능)
        return {
            "dong_code": payload.dong_code,
            "industry_code": payload.industry_code,
            "prediction": round(float(prediction_value), 1),
        }

    except Exception as e:
        # 예상치 못한 오류에 대한 로깅 및 처리
        print(f"An error occurred during prediction lookup: {e}")
        raise HTTPException(status_code=500, detail=f"예측 조회 중 오류 발생: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)