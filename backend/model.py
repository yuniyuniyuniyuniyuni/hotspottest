import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import json

MODEL_PATH = "model.joblib"
FEATURE_PATH = "feature_columns.json"

# 1) 데이터 로드
df = pd.read_csv("../data/predict_db/행정동X업종_통합_20221.csv")  # 경로 맞게 수정

# 2) 타겟 숫자화
df["생존률"] = pd.to_numeric(df["생존률"].replace({"-": None, "": None}), errors="coerce")

# 3) 식별자/이름 컬럼 드랍 (코드/이름 계열 전부 제거)
drop_cols = [c for c in df.columns if ("코드" in c) or c.endswith("_명")]
df = df.drop(columns=drop_cols, errors="ignore")
obj_ref_cols = df.select_dtypes(include="object").columns.tolist()

if "기준_년분기_코드" in df.columns:
    s = df["기준_년분기_코드"].astype(str)
    df["연도"] = s.str.extract(r"(20\d{2})").astype(float)
    df["분기"] = s.str.extract(r"(?:20\d{2})\D*([1-4])").astype(float)
    df = df.drop(columns=["기준_년분기_코드"])

# 4) 학습 가능한 행만 사용
df = df.dropna(subset=["생존률"])

# 5) 원-핫 인코딩 (남은 object 전부 인코딩)
obj_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)

# 6) X/y 분리
X = df.drop(columns=["생존률"])
y = df["생존률"]
feature_columns = X.columns.tolist()

# 7) 학습/검증 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 8) 모델 학습 (xgboost>=1.6이면 early_stopping_rounds 사용 가능)
model = XGBRegressor(
    tree_method="hist",       
    n_estimators=2000,        
    learning_rate=0.02,       
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,           
    reg_alpha=0.1,            
    min_child_weight=3,       
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=False)

joblib.dump(model, MODEL_PATH)
with open(FEATURE_PATH, 'w') as f:
    json.dump({"feature_columns": feature_columns, "obj_ref_cols": obj_ref_cols}, f)
    
pred = model.predict(X_val)
mse = mean_squared_error(y_val, pred)    
rmse = mse ** 0.5
print(f"RMSE: {rmse:.4f}")

r2 = r2_score(y_val, pred)
print(f"R2: {r2:.4f}")