import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import json
import numpy as np  # 안전한 나누기를 위해 numpy 추가
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.joblib"
FEATURE_PATH = "feature_columns.json"

# 1) 데이터 로드
# 경로가 현재 코드 위치 기준으로 되어 있으니, 실제 실행 환경에 맞게 확인해주세요.
df = pd.read_csv("../data/predict_db/train_서울시_2024_분기별.csv")
df = df[df['유사_업종_점포_수'] > 0]  # 점포 수가 0인 행 제거
df["점포당_매출_금액"] = df["당월_매출_금액"] / df["유사_업종_점포_수"]
categorical_features = ['행정동_코드_명', '서비스_업종_코드_명', '상권_변화_지표', '상권_변화_지표_명']
df = pd.get_dummies(df, columns=categorical_features, dummy_na=False)

obj_ref_cols = df.select_dtypes(include="object").columns.tolist() # 인코딩 전 object 컬럼 목록 저장

# 참고: 행정동, 업종과 같이 범주가 많은(고차원) 변수는 Target Encoding이나 CatBoost Encoding 사용 시 성능 향상 가능성이 높습니다.
obj_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)


# 6) X/y 분리
X = df.drop(columns=["점포당_매출_금액", '당월_매출_금액'], errors="ignore") 
y = np.log1p(df["점포당_매출_금액"])
feature_columns = X.columns.tolist()

# 7) 학습/검증 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. 찾은 상수 컬럼을 훈련 데이터와 검증 데이터 모두에서 제거합니다.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

SCALER_PATH = "scaler.joblib"
joblib.dump(scaler, SCALER_PATH)

# 8) 모델 학습
model = XGBRegressor(
    tree_method="hist",        # 속도 최적화
    n_estimators=1000,         # 충분히 큰 값으로 설정 (Early Stopping을 기대)
    learning_rate=0.01,        # 더 작게 설정하여 정밀한 학습 유도
    max_depth=10,               # 기존 6에서 약간 증가시켜 복잡도 조정
    subsample=0.85,            # 샘플링 비율 조정 (기존 0.8)
    colsample_bytree=0.85,     # 컬럼 샘플링 비율 조정 (기존 0.8)
    reg_lambda=3.0,            # L2 정규화 강화 (기존 2.0)
    reg_alpha=0.015,             # L1 정규화 강화 (기존 0.1)
    min_child_weight=5,        # 과적합 방지
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=50 # 학습 과정을 확인하기 위해 True로 변경
)

# 학습 완료 후, 최적의 n_estimators를 사용하여 모델 저장 (Early Stopping이 적용됨)
joblib.dump(model, MODEL_PATH)
with open(FEATURE_PATH, 'w') as f:
    json.dump({"feature_columns": feature_columns, "obj_ref_cols": obj_ref_cols}, f)
    
# 9) 평가
pred = model.predict(X_val_scaled)
mse = mean_squared_error(y_val, pred)
rmse = mse ** 0.5

pred_train = model.predict(X_train_scaled)
r2_train = r2_score(y_train, pred_train)

print("-" * 30)
print(f"RMSE: {rmse:.4f}")
r2 = r2_score(y_val, pred)
print(f"검증 데이터 R²: {r2:.4f}")
print(f"훈련 데이터 R²: {r2_train:.4f}")

# baseline_pred = np.full_like(y_val, y_train.mean())  # 모든 샘플에 훈련 타겟 평균값 예측
# baseline_mse = mean_squared_error(y_val, baseline_pred)
# baseline_rmse = baseline_mse ** 0.5
# baseline_r2 = r2_score(y_val, baseline_pred)

# print("-" * 30)
# print(f"[Baseline-Mean] RMSE: {baseline_rmse:.4f}")
# print(f"[Baseline-Mean] R2: {baseline_r2:.4f}")


# ----- 피쳐 중요도 시각화 -----
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib import rc

# # Mac 환경에서 한글 깨짐 방지 설정
# rc('font', family='AppleGothic')

# # 마이너스 부호 깨짐 방지 설정
# plt.rcParams['axes.unicode_minus'] = False

# # 피처 중요도를 데이터프레임으로 만들기
# feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
# feature_importances = feature_importances.sort_values('importance', ascending=False)

# # 상위 20개 피처 시각화
# plt.figure(figsize=(10, 8))
# sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
# plt.title('Top 20 Feature Importances')
# plt.show()
