import pandas as pd
import joblib
import json
import numpy as np
# main.py에 있던 함수들을 별도 파일로 분리하거나 가져와서 사용

print("배치 예측을 시작합니다...")

# 1. 리소스 로드 (모델, 스케일러, 학습/예측 데이터)
FEATURE_PATH = "feature_columns.json"
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
raw_df_2025 = pd.read_csv("../data/original_db/서울시_2025_2.csv")

categorical_features = ['행정동_코드_명', '서비스_업종_코드_명', '상권_변화_지표', '상권_변화_지표_명']
df = pd.get_dummies(raw_df_2025, columns=categorical_features, dummy_na=False)

obj_ref_cols = df.select_dtypes(include="object").columns.tolist() # 인코딩 전 object 컬럼 목록 저장

with open(FEATURE_PATH, 'r') as f:
    data = json.load(f)
    feature_columns = data['feature_columns']
    obj_ref_cols = data['obj_ref_cols']
            
# 참고: 행정동, 업종과 같이 범주가 많은(고차원) 변수는 Target Encoding이나 CatBoost Encoding 사용 시 성능 향상 가능성이 높습니다.
obj_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)
df["기준_년분기_코드"] = df["기준_년분기_코드"] % 10
X_2025 = df.reindex(columns=feature_columns, fill_value=0)

# 2. 예측할 데이터 전처리
X_2025_scaled = scaler.transform(X_2025)

# 3. 전체 데이터에 대한 예측 수행
all_predictions_log = model.predict(X_2025_scaled)
all_predictions_actual = np.expm1(all_predictions_log)

# 4. 예측 결과를 원본 데이터프레임에 추가
results_df = raw_df_2025.copy()
results_df['점포당_매출_금액_예측'] = all_predictions_actual


# 6. 최종 결과를 CSV 또는 다른 DB에 저장
output_path = "../data/predict_db/predictions_2025.csv"
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"배치 예측 완료! 결과가 {output_path} 에 저장되었습니다.")