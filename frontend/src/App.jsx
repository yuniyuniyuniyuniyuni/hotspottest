// App.jsx
import React, { useMemo, useState } from "react";
import "./App.css";

// 행정동 이름과 코드의 매핑 리스트를 불러옵니다.
// 이 파일은 사용자가 제공한 CSV 파일을 기반으로 생성되었다고 가정합니다.
import DONG_OPTIONS from "./dongOptions"; 

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

  // 행정동 이름 (사용자 입력)
  const [dongName, setDongName] = useState("");
  
  // 행정동 코드 (API 요청에 사용될 실제 값)
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

    // 1. 행정동 이름을 기반으로 코드를 찾습니다.
    const selectedDong = DONG_OPTIONS.find(
      (dong) => dong.name === dongName
    );

    let finalDongCode = dongCode;
    
    // 이름이 정확히 일치하는 코드가 있다면 업데이트
    if (selectedDong) {
      finalDongCode = selectedDong.code;
      setDongCode(finalDongCode); // 상태도 업데이트
    } 
    
    // 2. 입력값 및 코드 유효성 검사
    if (!finalDongCode || !industryCode) {
      // 코드를 찾지 못했거나 업종이 선택되지 않은 경우
      let msg = "";
      if (!finalDongCode) {
         msg += "유효한 행정동 이름을 입력했는지 확인해 주세요. ";
      }
      if (!industryCode) {
         msg += "업종을 선택해 주세요. ";
      }
      
      setErrorMsg(msg.trim() || "지역과 업종을 선택/입력해 주세요.");
      setLoading(false);
      return;
    }

    try {
      // 3. 찾은 코드로 API 요청
      const payload = { dong_code: finalDongCode, industry_code: industryCode };
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
    // CSV 업로드 로직은 변경 없이 유지
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
        // CSV 파일이 예측 결과를 여러 개 반환할 수 있지만, 여기서는 첫 번째 결과만 표시합니다.
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
          <h2>상권 이름 입력 후 예측</h2>
          <form className="form" onSubmit={handlePredictBySelection}>
            <div className="grid">
              <div className="field">
                <label htmlFor="dongNameInput">행정동 이름</label>
                {/* 행정동 이름 입력 필드 */}
                <input
                  id="dongNameInput"
                  type="text"
                  placeholder="예: 청운효자동" 
                  value={dongName}
                  onChange={(e) => setDongName(e.target.value)}
                  list="dong-names" // datalist와 연결
                />
                
                {/* 행정동 이름을 자동 완성할 수 있도록 datalist 추가 (선택 사항) */}
                <datalist id="dong-names">
                  {DONG_OPTIONS.map((dong) => (
                    <option key={dong.code} value={dong.name} />
                  ))}
                </datalist>
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
              {loading ? "예측 중..." : "예측하기"}
            </button>
          </form>
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