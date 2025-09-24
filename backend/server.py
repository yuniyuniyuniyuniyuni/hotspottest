from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "FastAPI 서버 동작 중!"}

@app.get("/predict")
def predict(region: str = "성수동", category: str = "카페"):
    # 더미 데이터 반환
    return {
        "region": region,
        "category": category,
        "success_rate": "72%",
        "risk_factors": ["높은 임대료", "경쟁과밀"],
        "success_factors": ["유동인구 많음", "배후 주거지 큼"]
    }

if __name__ == "__main__":
    # 여기서 바로 서버 실행 (uvicorn 내장 호출)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
