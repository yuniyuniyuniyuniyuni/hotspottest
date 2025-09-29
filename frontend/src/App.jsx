// App.jsx
import React, { useState } from "react";
import "./App.css";
import DONG_OPTIONS from "./dongOptions"; // 행정동 데이터
// 실제 서비스에서는 업종 데이터를 DB나 별도 파일로 관리할 것입니다.
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

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [selectedDongName, setSelectedDongName] = useState("");
  const [selectedIndustryCode, setSelectedIndustryCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showResults, setShowResults] = useState(false); // 결과 페이지 표시 여부

  // 결과 데이터 상태
  const [predictionResult, setPredictionResult] = useState(null); // 매출 예측 및 CBS 점수
  const [topIndustries, setTopIndustries] = useState([]); // 지역별 추천 업종
  const [topRegions, setTopRegions] = useState([]); // 업종별 추천 지역

  // 초기화 함수
  const resetApp = () => {
    setSelectedDongName("");
    setSelectedIndustryCode("");
    setLoading(false);
    setError("");
    setShowResults(false);
    setPredictionResult(null);
    setTopIndustries([]);
    setTopRegions([]);
  };

  // 모든 API를 호출하고 결과를 한 번에 받아오는 함수
  const analyzeCommercialArea = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    setShowResults(false); // 분석 시작 시 결과 숨김

    const dong = DONG_OPTIONS.find((d) => d.name === selectedDongName);
    if (!dong || !selectedIndustryCode) {
      setError("유효한 지역과 업종을 모두 선택해주세요.");
      setLoading(false);
      return;
    }

    const dongCode = dong.code;

    try {
      // 1. 상권 분석 (매출 및 CBS 점수)
      const predictResponse = await fetch(`${API_BASE}/predict_by_selection`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dong_code: dongCode, industry_code: selectedIndustryCode }),
      });
      if (!predictResponse.ok) throw new Error((await predictResponse.json()).detail || "상권 분석 실패");
      const predictData = await predictResponse.json();
      setPredictionResult(predictData);

      // 2. 지역별 추천 업종 (현재 선택된 지역 기준)
      const industriesResponse = await fetch(`${API_BASE}/recommend/industries?dong_code=${dongCode}`);
      if (!industriesResponse.ok) throw new Error((await industriesResponse.json()).detail || "업종 추천 실패");
      const industriesData = await industriesResponse.json();
      setTopIndustries(industriesData);

      // 3. 업종별 추천 지역 (현재 선택된 업종 기준)
      const regionsResponse = await fetch(`${API_BASE}/recommend/regions?industry_code=${selectedIndustryCode}`);
      if (!regionsResponse.ok) throw new Error((await regionsResponse.json()).detail || "지역 추천 실패");
      const regionsData = await regionsResponse.json();
      setTopRegions(regionsData);

      setShowResults(true); // 모든 API 호출 성공 시 결과 표시
    } catch (e) {
      setError(`분석 실패: ${e.message}`);
      setShowResults(false); // 오류 발생 시 결과 숨김
    } finally {
      setLoading(false);
    }
  };

  // 점수 게이지 스타일 계산
  const cbsScore = predictionResult?.cbs_score || 0;
  const progressStyle = {
    "--progress": `${cbsScore}%`,
    "--color":
      cbsScore >= 80 ? "#4CAF50" : cbsScore >= 60 ? "#FFC107" : "#F44336",
  };

  const selectedIndustryName = INDUSTRY_OPTIONS.find(
    (opt) => opt.code === selectedIndustryCode
  )?.name;

  return (
    <div className="app-container">
      {!showResults ? (
        // 초기 입력 폼 (Image 2)
        <div className="initial-form-card">
          <h1 className="title">장사잘될지도</h1>
          <form className="form" onSubmit={analyzeCommercialArea}>
            <div className="field">
              <label htmlFor="dongNameInput" className="label">
                지역
              </label>
              <input
                id="dongNameInput"
                type="text"
                placeholder="희망 지역을 선택하세요."
                value={selectedDongName}
                onChange={(e) => setSelectedDongName(e.target.value)}
                list="dong-names"
                className="input"
                required
              />
              <datalist id="dong-names">
                {DONG_OPTIONS.map((d) => (
                  <option key={d.code} value={d.name} />
                ))}
              </datalist>
            </div>
            <div className="field">
              <label htmlFor="industrySelect" className="label">
                업종
              </label>
              <select
                id="industrySelect"
                value={selectedIndustryCode}
                onChange={(e) => setSelectedIndustryCode(e.target.value)}
                className="input select"
                required
              >
                <option value="">희망 업종을 선택하세요.</option>
                {INDUSTRY_OPTIONS.map((opt) => (
                  <option key={opt.code} value={opt.code}>
                    {opt.name}
                  </option>
                ))}
              </select>
            </div>
            {error && <p className="error-message">{error}</p>}
            <button type="submit" className="analyze-button" disabled={loading}>
              {loading ? "분석 중..." : "성공률 분석"}
            </button>
          </form>
        </div>
      ) : (
        // 결과 대시보드 (Image 1)
        <div className="dashboard-container">
          <header className="dashboard-header">
            <h1 className="dashboard-title">장사잘될지도</h1>
            <div className="header-actions">
              <p className="selected-info">
                선택하신 지역은{" "}
                <span className="highlight-text">{selectedDongName}</span>,
                업종은 <span className="highlight-text">{selectedIndustryName}</span> 입니다.
              </p>
              <button className="pdf-button" onClick={resetApp}>
                재분석하기
              </button>
            </div>
          </header>

          <main className="dashboard-main">
            <div className="score-and-prediction">
              <div className="score-card card">
                <h3>점수</h3>
                <div className="progress-circle-container" style={progressStyle}>
                  <div className="progress-circle" data-progress={Math.round(cbsScore)}>
                    <span className="score-text">
                      {predictionResult ? Math.round(cbsScore) : "-"}점
                    </span>
                  </div>
                </div>
              </div>

              <div className="prediction-card card">
                <h3>매출 예측</h3>
                <p className="prediction-value">
                  {predictionResult
                    ? Number(predictionResult.prediction).toLocaleString()
                    : "-"}
                  원
                </p>
                <div className="strength-weakness">
                  {/* TODO: 강점/약점 요소는 백엔드 로직이 필요 */}
                  <label className="checkbox-container">
                    <input type="checkbox" checked={false} readOnly />
                    <span className="checkmark"></span>
                    강점 요소
                  </label>
                  <label className="checkbox-container">
                    <input type="checkbox" checked={false} readOnly />
                    <span className="checkmark"></span>
                    약점 요소
                  </label>
                </div>
              </div>
            </div>

            <div className="recommendations-grid">
              {/* 지역별 추천 업종 */}
              <div className="recommendation-card card">
                <h3>
                  <span className="highlight-text-small">{selectedDongName}</span>에서 성공확률 높은 업종 TOP 5
                </h3>
                <table className="recommendation-table">
                  <thead>
                    <tr>
                      <th>업종</th>

                      <th>점수</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topIndustries.length > 0 ? (
                      topIndustries.map((item, index) => (
                        <tr key={index}>
                          <td>{item.name}</td>

                          <td>{item.cbs_score.toFixed(1)}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="3">추천 업종이 없습니다.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* 업종별 추천 지역 */}
              <div className="recommendation-card card">
                <h3>
                  <span className="highlight-text-small">{selectedIndustryName}</span>으로 성공확률 높은 지역 TOP 5
                </h3>
                <table className="recommendation-table">
                  <thead>
                    <tr>
                      <th>지역</th>

                      <th>점수</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topRegions.length > 0 ? (
                      topRegions.map((item, index) => (
                        <tr key={index}>
                          <td>{item.name}</td>

                          <td>{item.cbs_score.toFixed(1)}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="3">추천 지역이 없습니다.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </main>
        </div>
      )}
    </div>
  );
}