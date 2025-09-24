from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 (React 프론트엔드와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서는 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "백엔드 FastAPI 서버 동작 중!"}

@app.get("/predict")
def predict(region: str = "성수동", category: str = "카페"):
    # ⚠️ 여기서는 임시 하드코딩 (모델 없이 더미 데이터 반환)
    return {
        "region": region,
        "category": category,
        "success_rate": "72%",
        "risk_factors": ["높은 임대료", "경쟁과밀"],
        "success_factors": ["유동인구 많음", "배후 주거지 큼"]
    }
