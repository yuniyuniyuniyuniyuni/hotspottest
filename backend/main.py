import pandas as pd
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv() 

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

def calculate_cbs_scores(df):
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
    
        # Step 3: 각 지표를 0-100점으로 정규화
        df['sales_norm'] = normalize(df['점포당_매출_금액_예측'])
        df['stability_norm'] = normalize(df['stability_index'])
        df['growth_norm'] = normalize(df['growth_index'])
        df['location_norm'] = normalize(df['location_advantage_index'])

        # Step 4: 정규화된 점수에 가중치를 적용하여 '원시' CBS 점수 계산
        df['cbs_raw_score'] = (
            df['sales_norm'] * 0.35 +
            df['stability_norm'] * 0.25 +
            df['growth_norm'] * 0.20 +
            df['location_norm'] * 0.20
        )        
        # ✨ Step 5 (NEW): 최종 CBS 점수를 다시 0-100으로 정규화
        df['cbs_score'] = normalize(df['cbs_raw_score'])
        print("✅ Step 5: 최종 CBS 점수 계산 완료")
        
        # 원시 점수는 더 이상 필요 없으므로 삭제 (선택 사항)
        df = df.drop(columns=['cbs_raw_score'])
        return df

    
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 데이터 로드 및 모든 점수 사전 계산 (정규화 로직 개선)"""
    global predictions_db, numeric_features
    try:
        df = pd.read_csv(PREDICTIONS_PATH)
        print(f"✅ Step 1: '{PREDICTIONS_PATH}'에서 원본 데이터 로드 완료 ({len(df)}개)")
        df.fillna(0, inplace=True)
        growth_map = {'상권확장': 1.5, '다이나믹': 1.2, '정체': 1.0, '상권축소': 0.8}
        df['상권_변화_가중치'] = df['상권_변화_지표'].map(growth_map)
        predictions_db = calculate_cbs_scores(df)
        
        numeric_features = predictions_db.select_dtypes(include=np.number).columns.tolist()
        print(f"✅ 숫자형 피처 {len(numeric_features)}개를 전역 변수에 저장했습니다.")
        
        print("🚀 서버가 성공적으로 시작되었습니다.")


    except Exception as e:
        print(f"❌ 오류: 서버 시작 중 예외 발생: {e}")
        predictions_db = None
    yield
    print("Application Shutdown.")

origins = [
    "http://localhost:5173",
]

# ===== 앱 생성, CORS, Pydantic 모델 (이전과 동일) =====
app = FastAPI(title="HotSpot API", version="0.4", description="상권 매출 예측 및 정규화된 CBS 기반 추천 API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
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

@app.get("/get_insight", summary="상권 분석 강점, 약점")
def get_insight(industry_code: str = Query(..., description="서비스 업종 코드"), dong_code: str = Query(..., description="행정동 코드")):
    import shap
    
    if predictions_db is None:
        raise HTTPException(status_code=503, detail="서버 리소스가 준비되지 않았습니다.")
    df = predictions_db.copy()
    df = df[(df['행정동_코드'] == int(dong_code)) & (df['서비스_업종_코드'] == industry_code)]
    if df.empty:
        raise HTTPException(status_code=404, detail="선택한 지역과 업종에 대한 데이터를 찾을 수 없습니다.")
    df.fillna(0, inplace=True)
    features_to_exclude = [
        '기준_년분기_코드', '행정동_코드', '서비스_업종_코드', '당월_매출_금액', '점포당_매출_금액',
        # CBS 계산에 사용된 중간 지표 및 최종 점수도 제외하는 것이 분석의 정확도를 높입니다.
        'stability_index', 'growth_index', 'location_advantage_index', 
        'sales_norm', 'stability_norm', 'growth_norm', 'location_norm', 'cbs_score'
    ]
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = [f for f in numeric_cols if f not in features_to_exclude]
    X = df[features]
    
    if X.empty or len(X.columns) == 0:
        raise HTTPException(status_code=500, detail="분석할 수치 데이터를 찾을 수 없습니다.")

    def calculate_cbs_for_shap_local(X_values: np.ndarray) -> np.ndarray:
        # 컬럼 이름으로 전역 변수 대신 현재 분석 대상인 'X.columns'를 사용합니다.
        row_df = pd.DataFrame(X_values, columns=X.columns)
        
        epsilon = 1e-6 # 0으로 나누는 것을 방지
        
        # seoul_avg_op_months가 없을 수 있으므로 전체 데이터베이스의 평균을 사용하거나 안전한 기본값을 설정
        seoul_avg_op_months = predictions_db['운영_영업_개월_평균'].mean()  # type: ignore
        if seoul_avg_op_months == 0: seoul_avg_op_months = 1

        stability_index = (1 - row_df['폐업_률']) * 100 * (row_df['운영_영업_개월_평균'] / seoul_avg_op_months)
        growth_index = (row_df['개업_율'] / (row_df['폐업_률'] + epsilon)) * row_df['상권_변화_가중치'] * 100
        locational_advantage_index = (row_df['총_유동인구_수'] / (row_df['점포_수'] + epsilon)) * (row_df['총_직장_인구_수'] / 10000) * 0.1
        predicted_sales = row_df['점포당_매출_금액_예측']
        
        cbs_scores = (predicted_sales * 0.35) + (stability_index * 0.25) + (growth_index * 0.20) + (locational_advantage_index * 0.20)
        return cbs_scores.to_numpy() # 결과를 numpy 배열로 반환

    instance_to_explain = X.iloc[[0]]
    # 수정된 local 함수를 Explainer에 전달합니다.
    explainer = shap.Explainer(calculate_cbs_for_shap_local, X)
    shap_values = explainer(instance_to_explain)
    
    results_df = pd.DataFrame({
        'Feature': X.columns,
        'Actual_Value': instance_to_explain.iloc[0],
        'Mean_Value': X.mean(),
        'SHAP_Value': shap_values.values[0]
    }).sort_values(by='SHAP_Value', ascending=False)
    strengths = []
    weaknesses = []
    shap_result_text = f"""- 분석 대상: {dong_code} / {industry_code}
    - 기본 점수(평균 CBS): {shap_values.base_values[0]:,.0f}
    - 최종 예측 CBS 점수: {explainer.model(instance_to_explain.values)[0]:,.0f}
    - 강점 Top 5 (점수를 올린 요인):
    """
    
    for _, row in results_df.head(5).iterrows():
        direction = "평균보다 높음" if row['Actual_Value'] > row['Mean_Value'] else "평균보다 낮음"
        strengths.append(f"  - {row['Feature']}: {row['Actual_Value']:,.2f} ({direction})\n")
        shap_result_text += f"  - {row['Feature']}: {row['Actual_Value']:,.2f} ({direction})\n"
        
    shap_result_text += "- 약점 Top 5 (점수를 내린 요인):\n"
    for _, row in results_df.tail(5).sort_values(by='SHAP_Value', ascending=True).iterrows():
        direction = "평균보다 높음" if row['Actual_Value'] > row['Mean_Value'] else "평균보다 낮음"
        weaknesses.append(f"  - {row['Feature']}: {row['Actual_Value']:,.2f} ({direction})\n")
        shap_result_text += f"  - {row['Feature']}: {row['Actual_Value']:,.2f} ({direction})\n"

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "shap_result_text": shap_result_text,
    }
    
@app.get("/ai_insight", summary="AI 기반 상권 분석 리포트")
def ai_insight(industry_code: str = Query(..., description="서비스 업종 코드"), dong_code: str = Query(..., description="행정동 코드")):
    insight = get_insight(industry_code, dong_code)
    shap_result_text = insight["shap_result_text"]
    prompt = f"""
    당신은 대한민국 최고의 상권분석 전문가입니다. 예비 창업자에게 조언하는 역할입니다.
    아래 분석 데이터를 바탕으로, 전문적이지만 이해하기 쉬운 최종 컨설팅 의견을 작성해주세요.

    [분석 데이터]
    {shap_result_text}

    [작성 가이드라인]
    1. **결론 요약:** 이 상권의 핵심 특징과 기회/위험 요인을 한두 문장으로 요약합니다.
    2. **강점 분석:** 점수를 올린 가장 중요한 요인 2~3가지를 선택해,
    - (a) 구체적인 데이터 지표 설명 →
    - (b) 해당 지표가 창업자에게 어떤 의미가 있는지 해석 →
    - (c) 실제 전략적 시사점 제시
    의 구조로 각각 서술합니다.
    3. **약점 분석:** 점수를 내린 가장 중요한 요인 2~3가지를 선택해,
    동일하게 (a) 지표 → (b) 의미 → (c) 시사점 구조로 자세히 분석합니다.
    4. **최종 전략 제언:** 위 분석을 종합하여, 이 상권에 진입하려는 예비 창업자에게 구체적이고 실행 가능한 조언을 한두 문장으로 제시합니다.
    5. 답변은 반드시 한글로, 친절하고 전문가적인 톤으로 작성해주세요.
    """

    # ★★★ 최종 수정된 부분 ★★★
    # --- OpenAI API 호출 ---
    try:
        # os.environ.get()을 사용하여 환경 변수에서 API 키를 안전하게 가져옵니다.
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            # 환경 변수에 키가 없는 경우에만 경고 메시지를 보여줍니다.
            ai_interpretation = "OpenAI API 키가 환경 변수에 설정되지 않았습니다. 터미널에서 'export OPENAI_API_KEY=\"sk-...\"' 명령어를 실행해주세요."
        else:
            client = OpenAI(api_key=api_key) # type: ignore
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 대한민국 최고의 상권분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
            )
            ai_interpretation = response.choices[0].message.content
            print("✅ 7. AI 컨설턴트 분석을 성공적으로 생성했습니다.")

    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        ai_interpretation = f"API 호출 중 오류가 발생하여 자동 해석을 생성하지 못했습니다: {e}"
        
    return {
        "report": ai_interpretation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)