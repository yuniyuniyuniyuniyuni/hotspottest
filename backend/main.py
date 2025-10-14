import pandas as pd
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from dotenv import load_dotenv
import json
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
        print("✅ 최종 CBS 점수 계산 완료")
        
        # 원시 점수는 더 이상 필요 없으므로 삭제 (선택 사항)
        df = df.drop(columns=['cbs_raw_score'])
        return df

    
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 데이터 로드 및 모든 점수 사전 계산 (정규화 로직 개선)"""
    global predictions_db, numeric_features
    try:
        df = pd.read_csv(PREDICTIONS_PATH)
        print(f"✅ '{PREDICTIONS_PATH}'에서 원본 데이터 로드 완료 ({len(df)}개)")
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
        "cbs_score": round(float(data['cbs_score']), 2), # 미리 계산된 CBS 점수 반환
        "rent": round(float(data['임대료']), 2)
    }

@app.get("/rent_distribution", summary="지역 내 임대료 분포 데이터 조회")
def get_rent_distribution(dong_code: str = Query(..., description="행정동 코드"), industry_code: str = Query(..., description="서비스 업종 코드")):
    if predictions_db is None:
        raise HTTPException(status_code=503, detail="서버 리소스가 준비되지 않았습니다.")

    # 1. 해당 지역(동)의 모든 임대료 데이터 추출
    dong_df = predictions_db[predictions_db['행정동_코드'] == int(dong_code)]
    if dong_df.empty:
        raise HTTPException(status_code=404, detail=f"'{dong_code}' 지역 데이터를 찾을 수 없습니다.")

    all_rents = predictions_db['임대료'].dropna()
    if all_rents.empty:
        return {
            "bins": [], "counts": [], "current_rent_bin_index": -1,
            "current_rent": 0, "top_percentile": 50 # 데이터 없을 시 중간값 반환
        }

    # 2. 현재 선택한 업종의 특정 임대료 조회
    current_selection_df = dong_df[dong_df['서비스_업종_코드'] == industry_code]
    current_rent = current_selection_df.iloc[0].get('임대료', 0) if not current_selection_df.empty else 0

    # 3. ✨ 신규: 상위 퍼센트 계산
    if len(all_rents) > 0:
        # 자신보다 임대료가 낮은 업종의 수 계산
        count_lower = (all_rents < current_rent).sum()
        # 하위 백분위 계산 (0~100)
        percentile_from_bottom = (count_lower / len(all_rents)) * 100
        # 상위 백분위로 변환 (e.g., 하위 90% -> 상위 10%)
        top_percentile = 100.0 - percentile_from_bottom
    else:
        top_percentile = 50.0 # 데이터가 없는 경우를 위한 기본값

    # 4. 임대료 데이터를 10개 구간으로 나누어 히스토그램 생성
    counts, bins = np.histogram(all_rents, bins=10)

    # 5. 현재 임대료가 어느 구간에 속하는지 인덱스 찾기
    current_rent_bin_index = np.digitize(current_rent, bins) - 1
    if current_rent_bin_index == len(counts):
        current_rent_bin_index -= 1

    return {
        "bins": bins.tolist(),
        "counts": counts.tolist(),
        "current_rent_bin_index": int(current_rent_bin_index),
        "current_rent": int(current_rent),
        "top_percentile": round(top_percentile, 1) # 계산된 상위 퍼센트 값을 응답에 추가
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
                           cbs_score=round(row['cbs_score'], 2),
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
                           cbs_score=round(row['cbs_score'], 2),
                           store_count=int(row['점포_수']) if pd.notna(row['점포_수']) else 0)
        for _, row in top_5.iterrows()
    ]

@app.get("/get_insight", summary="상권 분석 강점, 약점")
def get_insight(industry_code: str = Query(..., description="서비스 업종 코드"), dong_code: str = Query(..., description="행정동 코드")):
    import shap
    
    if predictions_db is None:
        raise HTTPException(status_code=503, detail="서버 리소스가 준비되지 않았습니다.")

    # --- 1. 분석에 사용할 피처(feature) 목록 정의 ---
    features_to_exclude = [
        '기준_년분기_코드', '행정동_코드', '서비스_업종_코드', '당월_매출_금액', '점포당_매출_금액',
        'stability_index', 'growth_index', 'location_advantage_index', 
        'sales_norm', 'stability_norm', 'growth_norm', 'location_norm', 'cbs_score',
        '엑스좌표_값', '와이좌표_값', '행정동_코드_명', '서비스_업종_코드_명', '상권_변화_지표', '상권_변화_지표_명'
    ]
    all_numeric_features = predictions_db.select_dtypes(include=np.number).columns.tolist()
    features = [f for f in all_numeric_features if f not in features_to_exclude]

    # --- 2. 배경 데이터(전체)와 분석 대상 데이터(특정 행) 분리 ---
    background_data = predictions_db[features].fillna(0)
    instance_df = predictions_db[
        (predictions_db['행정동_코드'] == int(dong_code)) & 
        (predictions_db['서비스_업종_코드'] == industry_code)
    ]
    cbs_features = ['점포당_매출_금액_예측', '서울_운영_영업_개월_평균', '폐업_률', '운영_영업_개월_평균', '개업_율', '상권_변화_가중치', '총_유동인구_수', '점포_수', '총_직장_인구_수']
    background_data_cbs = background_data[cbs_features]
    if instance_df.empty:
        raise HTTPException(status_code=404, detail="선택한 지역과 업종에 대한 데이터를 찾을 수 없습니다.")

    instance_to_explain_cbs = instance_df[cbs_features].fillna(0)
    instance_to_explain = instance_df[features].fillna(0)
    if instance_to_explain.empty:
        raise HTTPException(status_code=500, detail="분석할 수치 데이터를 찾을 수 없습니다.")

    # --- 3. SHAP 분석을 위한 모델 함수 정의 (수정 없음) ---
    def calculate_cbs_for_shap_local(X_values: np.ndarray) -> np.ndarray:
        row_df = pd.DataFrame(X_values, columns=features)
        epsilon = 1e-6
        seoul_avg_op_months = row_df['서울_운영_영업_개월_평균']
        stability_index = (1 - row_df['폐업_률']) * 100 * (row_df['운영_영업_개월_평균'] / (seoul_avg_op_months + epsilon))
        growth_index = (row_df['개업_율'] / (row_df['폐업_률'] + epsilon)) * row_df['상권_변화_가중치'] * 100
        locational_advantage_index = (row_df['총_유동인구_수'] / (row_df['점포_수'] + epsilon)) * (row_df['총_직장_인구_수'] / 10000) * 0.1
        predicted_sales = row_df['점포당_매출_금액_예측']
        cbs_scores = (predicted_sales * 0.35) + (stability_index * 0.25) + (growth_index * 0.20) + (locational_advantage_index * 0.20)
        return cbs_scores.to_numpy()

    # --- 4. SHAP 분석 실행 ---
    cbs_explainer = shap.Explainer(calculate_cbs_for_shap_local, background_data_cbs)
    sales_explainer = shap.Explainer(calculate_cbs_for_shap_local, background_data)
    shap_values_cbs = cbs_explainer(instance_to_explain_cbs)
    shap_values_sales = sales_explainer(instance_to_explain)

    # --- 5. 분석 결과 정리 및 텍스트 리포트 생성 (### 수정된 부분 ###) ---

    # 분석 대상인스턴의 SHAP 값 배열
    cbs_values_instance = shap_values_cbs.values[0]
    sales_values_instance = shap_values_sales.values[0]

    # 영향력의 총합(절대값 기준) 계산
    total_impact = np.abs(cbs_values_instance).sum()
    epsilon = 1e-6 

    cbs_results_df = pd.DataFrame({
        'Feature': cbs_features,
        'Actual_Value': instance_to_explain_cbs.iloc[0].values,
        'Mean_Value': background_data_cbs.mean().values,
        'SHAP_Value': cbs_values_instance,
        'Contribution_Percent': (cbs_values_instance / (total_impact + epsilon)) * 100
    }).sort_values(by='SHAP_Value', key=abs, ascending=False).reset_index(drop=True)

    sales_results_df = pd.DataFrame({
        'Feature': background_data.columns,
        'Actual_Value': instance_to_explain.iloc[0],
        'Mean_Value': background_data.mean(),
        'SHAP_Value': shap_values_sales.values[0]
    }).sort_values(by='SHAP_Value', ascending=False)
    
    base_score = shap_values_cbs.base_values[0]
    predicted_score = base_score + cbs_results_df['SHAP_Value'].sum()

    strengths = []
    weaknesses = []
    cbs = []

    dong_name = instance_df.iloc[0]['행정동_코드_명']
    industry_name = instance_df.iloc[0]['서비스_업종_코드_명']

    shap_result_text = f"""- 분석 대상: {dong_name} / {industry_name}
    - 기본 점수(전체 상권 평균): {base_score:,.0f}점
    - 최종 예측 점수: {predicted_score:,.0f}점
    - cbs 결정 요인
    """

    for _, row in cbs_results_df.head(5).iterrows():
        direction = "높음" if row['Actual_Value'] > row['Mean_Value'] else "낮음"
        cbs_detail = (f"{row['Feature']}: {row['Actual_Value']:,.0f} "
                           f"(평균값: {row['Mean_Value']:,.0f}, 영향력: {row['Contribution_Percent']:.1f}%)")
        shap_result_text += "\n- " + cbs_detail
        cbs.append(cbs_detail)

    shap_result_text += "- 강점 Top 5 (매출을 올린 요인) -"
    for _, row in sales_results_df.head(5).iterrows():
        direction = "높음" if row['Actual_Value'] > row['Mean_Value'] else "낮음"
        sales_detail = (f"{row['Feature']}: {row['Actual_Value']:,.0f}  (평균값: {row['Mean_Value']:,.0f}) ")
        shap_result_text += "\n- " + sales_detail
        strengths.append(sales_detail)
        
    shap_result_text += "- 약점 Top 5 (매출을 낮춘 요인) -"
    for _, row in sales_results_df.tail(5).iterrows():
        direction = "높음" if row['Actual_Value'] > row['Mean_Value'] else "낮음"
        sales_detail = (f"{row['Feature']}: {row['Actual_Value']:,.0f}  (평균값: {row['Mean_Value']:,.0f}) ")
        shap_result_text += "\n- " + sales_detail
        weaknesses.append(sales_detail)

    return {
        "dong_name": dong_name,
        "industry_name": industry_name,
        "cbs": cbs,
        "strengths": strengths,
        "weaknesses": weaknesses,
        # shap_result_text는 이제 프롬프트에 직접 사용되지 않지만, 디버깅 등을 위해 유지
        "shap_result_text": shap_result_text 
    }
    
@app.get("/ai_insight", summary="AI 기반 상권 분석 리포트")
def ai_insight(industry_code: str = Query(..., description="서비스 업종 코드"), dong_code: str = Query(..., description="행정동 코드")):
    insight_data = get_insight(industry_code, dong_code)
    
    def format_list_for_prompt(items: list) -> str:
        """리스트의 각 항목을 줄바꿈 문자로 연결하여 하나의 문자열로 만듭니다."""
        return "\n".join([f"- {item}" for item in items])

    cbs_factors_str = format_list_for_prompt(insight_data["cbs"])
    strengths_str = format_list_for_prompt(insight_data["strengths"])
    weaknesses_str = format_list_for_prompt(insight_data["weaknesses"])

    prompt = f"""
    당신은 대한민국 최고의 상권분석 전문가입니다. 예비 창업자에게 조언하는 역할입니다.
    아래 [분석 정보]와 [핵심 분석 데이터]를 바탕으로, 전문적이지만 이해하기 쉬운 최종 컨설팅 의견을 작성해주세요.

    [분석 정보]
    - 분석 지역: {insight_data["dong_name"]}
    - 분석 업종: {insight_data["industry_name"]}

    [핵심 분석 데이터]
    1. CBS 점수 결정 요인 (영향력 순):
    {cbs_factors_str}

    2. 예상 매출에 긍정적 영향을 준 요인 (강점):
    {strengths_str}

    3. 예상 매출에 부정적 영향을 준 요인 (약점):
    {weaknesses_str}

    [작성 가이드라인]
    1. **결론 요약 (summary):** [분석 정보]와 [핵심 분석 데이터]를 종합하여 이 상권의 핵심 특징, 기회, 위험 요인을 한두 문장으로 요약합니다.
    2. **CBS 결정 요인 분석 (cbs_analysis):** [핵심 분석 데이터]의 'CBS 점수 결정 요인' 중 상위 3가지가 이 상권의 종합적인 매력도에 어떤 영향을 미치는지 이유를 들어 설명합니다.
    3. **강점 및 약점 평가 (evaluation):** [핵심 분석 데이터]의 '강점'과 '약점' 데이터를 각각 2~3가지씩 활용하여, 실제 창업 시 어떤 점을 활용하고 어떤 점을 보완해야 할지 분석합니다.
    4. **최종 전략 제언 (strategy):** 모든 분석을 종합하여, 이 상권에 진입하려는 예비 창업자에게 구체적이고 실행 가능한 조언을 한두 문장으로 제시합니다.
    5. 답변은 반드시 한글로, 친절하고 전문가적인 톤으로 작성해주세요.
    
    [JSON 출력 규칙]
    - 최종 결과는 반드시 "summary", "cbs_analysis", "evaluation", "strategy" 키를 포함하는 JSON 형식으로만 응답해야 합니다.
    - **매우 중요**: 각 키에 해당하는 값(value)은 반드시 여러 문장으로 구성된 단일 텍스트 문자열(a single string)이어야 합니다.
    - **절대로 값 부분에 JSON 객체나 리스트(`{{}}`, `[]`)를 중첩하여 사용하지 마세요.**
    """

    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            return {
                "report": {
                    "summary": "AI 분석 실패",
                    "cbs_analysis": "OpenAI API 키가 설정되지 않았습니다.",
                    "evaluation": "서버 환경 변수를 확인해주세요.",
                    "strategy": ""
                }
            }

        client = OpenAI(api_key=api_key)
        
        # ★★★ API 호출 수정: JSON 모드 활성화 ★★★
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a top commercial district analyst in South Korea. Your response must be in JSON object format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        # OpenAI 응답이 JSON 문자열이므로 파싱해서 반환
        ai_response_dict = json.loads(response.choices[0].message.content) #type: ignore
        keys = ["summary", "cbs_analysis", "evaluation", "strategy"]
        default_message = "AI가 해당 항목에 대한 분석을 생성하지 못했습니다."
        
        ai_interpretation = {key: ai_response_dict.get(key, default_message) for key in keys}
        print("✅ AI 컨설턴트 분석을 성공적으로 생성했습니다.")

    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        ai_interpretation = {
            "summary": "AI 분석 중 오류 발생",
            "cbs_analysis": f"오류: {e}",
            "evaluation": "잠시 후 다시 시도해주세요.",
            "strategy": ""
        }
        
    return {
        "report": ai_interpretation
    }

@app.get("/stats", summary="주요 통계 조회")
def get_stats(dong_code: str = Query(..., description="행정동 코드"), industry_code: str = Query(..., description="서비스 업종 코드")):
    if predictions_db is None:
        raise HTTPException(status_code=503, detail="서버 리소스가 준비되지 않았습니다.")

    # 1. 서울시 전체의 평균 CBS 점수
    avg_cbs_score_seoul = predictions_db['cbs_score'].mean()

    # 2. 선택된 지역(동)의 모든 업종 평균 매출
    dong_df = predictions_db[predictions_db['행정동_코드'] == int(dong_code)]
    avg_sales_dong = dong_df['점포당_매출_금액_예측'].mean() if not dong_df.empty else 0

    # 3. 선택된 업종의 모든 지역 평균 매출
    industry_df = predictions_db[predictions_db['서비스_업종_코드'] == industry_code]
    avg_sales_industry = industry_df['점포당_매출_금액_예측'].mean() if not industry_df.empty else 0

    return {
        "avg_cbs_score_seoul": round(avg_cbs_score_seoul, 1),
        "avg_sales_dong": round(avg_sales_dong),
        "avg_sales_industry": round(avg_sales_industry)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)