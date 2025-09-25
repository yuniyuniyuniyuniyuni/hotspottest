// App.jsx
import React, { useMemo, useState } from "react";
import "./App.css";
import DONG_OPTIONS from "./dongOptions.js";

const DEFAULT_API_BASE =
  import.meta.env.VITE_API_BASE ||
  "http://localhost:8000";

const INDUSTRY_OPTIONS = [
  { code: "CS100001", name: "한식음식점" },
  { code: "CS100002", name: "중식음식점" },
  { code: "CS100003", name: "일식음식점" },
  { code: "CS100004", name: "양식음식점" },
  { code: "CS100005", name: "제과점" },
  { code: "CS100006", name: "패스트푸드점" },
  { code: "CS100007", name: "치킨전문점" },
  { code: "CS100008", name: "분식전문점" },
  { code: "CS100009", name: "호프-간이주점" },
  { code: "CS100010", name: "커피-음료" },
];

export default function App() {
  const API_BASE = useMemo(() => DEFAULT_API_BASE, []);

  const [dongCode, setDongCode] = useState("");
  const [industryCode, setIndustryCode] = useState("");

  const [file, setFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [predictionValue, setPredictionValue] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  const handlePredictBySelection = async (e) => {
    e.preventDefault();
    setErrorMsg("");
    setLoading(true);
    setPredictionValue(null);

    if (!dongCode || !industryCode) {
      setErrorMsg("지역과 업종을 선택해 주세요.");
      setLoading(false);
      return;
    }

    try {
      const payload = { dong_code: dongCode, industry_code: industryCode };
      const response = await fetch(`${API_BASE}/predict_by_selection`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP error! status: ${response.status}, detail: ${errorData.detail}`);
      }

      const data = await response.json();
      setPredictionValue(data.prediction);
    } catch (e) {
      console.error("Prediction failed:", e);
      setErrorMsg(`예측 요청 실패: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCSVUpload = async () => {
    setErrorMsg("");
    setLoading(true);
    setPredictionValue(null);

    if (!file) {
      setErrorMsg("파일을 선택해 주세요.");
      setLoading(false);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/predict_csv`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (Array.isArray(data) && data.length > 0) {
        setPredictionValue(data[0].prediction);
      } else {
        setErrorMsg("CSV 예측 결과가 올바르지 않습니다.");
      }
    } catch (e) {
      console.error("CSV upload failed:", e);
      setErrorMsg(`CSV 업로드 및 예측 실패: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app__header">
        <h1>✨ 상권 생존율 예측</h1>
      </header>
      <main className="app__main">
        <section className="card">
          <h2>상권 선택 후 예측</h2>
          <form className="form" onSubmit={handlePredictBySelection}>
            <div className="grid">
              <div className="field">
                <label>행정동 코드</label>
                <select
                  value={dongCode}
                  onChange={(e) => setDongCode(e.target.value)}
                >
                  <option value="">지역 선택</option>
                  {DONG_OPTIONS.map(option => (
                    <option key={option.code} value={option.code}>{option.name}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>서비스 업종 코드</label>
                <select
                  value={industryCode}
                  onChange={(e) => setIndustryCode(e.target.value)}
                >
                  <option value="">업종 선택</option>
                  {INDUSTRY_OPTIONS.map(option => (
                    <option key={option.code} value={option.code}>{option.name}</option>
                  ))}
                </select>
              </div>
            </div>
            <button className="btn" type="submit" disabled={loading}>
              {loading ? "예측 중..." : "선택하여 예측하기"}
            </button>
          </form>
        </section>

        <section className="card">
          <h2>CSV 일괄 예측</h2>
          <div className="csv">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <button className="btn btn--ghost" onClick={handleCSVUpload} disabled={loading}>
              업로드 & 예측
            </button>
          </div>
        </section>

        {errorMsg && (
          <section className="card card--error">
            <h3>오류</h3>
            <pre className="pre">{errorMsg}</pre>
          </section>
        )}

        {predictionValue !== null && (
          <section className="card card--success">
            <h2>예측 결과</h2>
            <div className="prediction-value-container">
              {/* 백엔드에서 받은 값을 그대로 표시 */}
              <span className="prediction-value">
                {Number(predictionValue).toFixed(2)}%
              </span>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <span>MVP • HotSpot</span>
      </footer>
    </div>
  );
}