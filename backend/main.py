import pandas as pd
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# --- 전역 변수 ---
predictions_db: Optional[pd.DataFrame] = None
PREDICTIONS_PATH = "../data/predict_db/predictions_2025.csv"

# --- CBS 계산을 위한 함수들 (수정 없음) ---

def map_commercial_change_indicator(indicator_name: str) -> int:
    mapping = {'상권축소': 1, '정체': 2, '활성화': 3, '다이나믹': 4}
    return mapping.get(indicator_name, 1)

def normalize(series: pd.Series) -> pd.Series:
    """Pandas Series를 0-100 사이 값으로 정규화합니다."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series(50, index=series.index)
    return 100 * (series - min_val) / (max_val - min_val)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 데이터 로드 및 모든 점수 사전 계산 (정규화 로직 개선)"""
    global predictions_db
    try:
        df = pd.read_csv(PREDICTIONS_PATH)
        print(f"✅ Step 1: '{PREDICTIONS_PATH}'에서 원본 데이터 로드 완료 ({len(df)}개)")

        # Step 2: CBS 각 구성 지표를 계산하여 새 컬럼으로 추가
        seoul_avg_op_months = df['서울_운영_영업_개월_평균']
        df['stability_index'] = (1 - df['폐업_률']) * 100 * (df['운영_영업_개월_평균'] / seoul_avg_op_months)
        change_indicator_vals = df['상권_변화_지표_명'].apply(map_commercial_change_indicator)
        df['growth_index'] = np.where(
            df['폐업_률'] == 0, 0,
            (df['개업_율'] / df['폐업_률']) * change_indicator_vals * 100
        )
        df['location_advantage_index'] = np.where(
            df['점포_수'] == 0, 0,
            (df['총_유동인구_수'] / df['점포_수']) * (df['총_직장_인구_수'] / 10000) * 0.1
        )
        print("✅ Step 2: 안정성, 성장성, 입지 우위 지수 계산 완료")

        # Step 3: 각 지표를 0-100점으로 정규화
        df['sales_norm'] = normalize(df['점포당_매출_금액_예측'])
        df['stability_norm'] = normalize(df['stability_index'])
        df['growth_norm'] = normalize(df['growth_index'])
        df['location_norm'] = normalize(df['location_advantage_index'])
        print("✅ Step 3: 모든 지표 0-100점 스케일로 정규화 완료")

        # Step 4: 정규화된 점수에 가중치를 적용하여 '원시' CBS 점수 계산
        df['cbs_raw_score'] = (
            df['sales_norm'] * 0.35 +
            df['stability_norm'] * 0.25 +
            df['growth_norm'] * 0.20 +
            df['location_norm'] * 0.20
        )
        print("✅ Step 4: 가중치 적용된 원시 CBS 점수 계산 완료")
        
        # ✨ Step 5 (NEW): 최종 CBS 점수를 다시 0-100으로 정규화
        df['cbs_score'] = normalize(df['cbs_raw_score'])
        print("✅ Step 5: 최종 CBS 점수를 0-100 스케일로 재정규화 완료")
        
        # 원시 점수는 더 이상 필요 없으므로 삭제 (선택 사항)
        df = df.drop(columns=['cbs_raw_score'])

        predictions_db = df
        print("🚀 서버가 성공적으로 시작되었습니다.")

    except Exception as e:
        print(f"❌ 오류: 서버 시작 중 예외 발생: {e}")
        predictions_db = None
    yield
    print("Application Shutdown.")

# ===== 앱 생성, CORS, Pydantic 모델 (이전과 동일) =====
app = FastAPI(title="HotSpot API", version="0.4", description="상권 매출 예측 및 정규화된 CBS 기반 추천 API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
class RecommendationItem(BaseModel):
    name: str; code: str; cbs_score: float; store_count: int 
class PredictSelectionPayload(BaseModel):
    dong_code: str; industry_code: str

# ===== API 라우터 (훨씬 단순해짐) =====
@app.get("/")
def health_check():
    return {"status": "ok", "prediction_data_ready": predictions_db is not None}

@app.post("/predict_by_selection", summary="선택한 상권 매출 및 CBS 점수 조회")
def predict_by_selection(payload: PredictSelectionPayload):
    if predictions_db is None:
        raise HTTPException(status_code=503, detail="서버 리소스(예측 DB)가 준비되지 않았습니다.")
    
    result_row = predictions_db[
        (predictions_db['행정동_코드'] == int(payload.dong_code)) &
        (predictions_db['서비스_업종_코드'] == payload.industry_code)
    ]
    if result_row.empty:
        raise HTTPException(status_code=404, detail="선택한 지역과 업종에 대한 데이터를 찾을 수 없습니다.")
    
    data = result_row.iloc[0]
    return {
        "dong_code": payload.dong_code, 
        "industry_code": payload.industry_code,
        "prediction": round(float(data['점포당_매출_금액_예측']), 1),
        "cbs_score": round(float(data['cbs_score']), 2) # 미리 계산된 CBS 점수 반환
    }

@app.get("/recommend/regions", summary="업종별 최적 지역 Top 5 추천", response_model=List[RecommendationItem])
def get_top_regions_for_industry(industry_code: str = Query(..., description="서비스 업종 코드")):
    if predictions_db is None:
        raise HTTPException(status_code=503, detail="서버 리소스가 준비되지 않았습니다.")

    filtered_df = predictions_db[predictions_db['서비스_업종_코드'] == industry_code]
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail=f"'{industry_code}' 업종 데이터를 찾을 수 없습니다.")

    # 미리 계산된 cbs_score로 정렬만 수행
    top_5 = filtered_df.sort_values(by='cbs_score', ascending=False).head(5)
    
    return [
        RecommendationItem(name=row['행정동_코드_명'], 
                           code=str(row['행정동_코드']), 
                           cbs_score=row['cbs_score'], 
                           store_count=int(row['점포_수']) if pd.notna(row['점포_수']) else 0) 
        for _, row in top_5.iterrows()
    ]

@app.get("/recommend/industries", summary="지역별 최적 업종 Top 5 추천", response_model=List[RecommendationItem])
def get_top_industries_for_region(dong_code: str = Query(..., description="행정동 코드")):
    if predictions_db is None:
        raise HTTPException(status_code=503, detail="서버 리소스가 준비되지 않았습니다.")
    
    filtered_df = predictions_db[predictions_db['행정동_코드'] == int(dong_code)]
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail=f"'{dong_code}' 지역 데이터를 찾을 수 없습니다.")

    # 미리 계산된 cbs_score로 정렬만 수행
    top_5 = filtered_df.sort_values(by='cbs_score', ascending=False).head(5)

    return [
        RecommendationItem(name=row['서비스_업종_코드_명'], 
                           code=row['서비스_업종_코드'], 
                           cbs_score=row['cbs_score'], 
                           store_count=int(row['점포_수']) if pd.notna(row['점포_수']) else 0)
        for _, row in top_5.iterrows()
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)