// src/App.jsx
import React, { useMemo, useState } from "react";
import "./App.css";
import DONG_OPTIONS from "./dongOptions";

/** 업종 목록 (MVP용) */
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

const API_BASE = import.meta.env?.VITE_API_BASE || "http://localhost:8000";
const MAIN_PURPLE = "#8A60E6";

// AI 응답 텍스트의 줄바꿈을 처리하기 위한 작은 컴포넌트
const ParsedText = ({ text }) => {
  if (!text) return null;
  // \n을 기준으로 문단을 나누고, 각 문단을 <p> 태그로 감쌉니다.
  return text.split('\n').map((paragraph, index) => (
    <p key={index} className="strategy-paragraph">
      {paragraph || <br />}
    </p>
  ));
};

export default function App() {
  const [dongName, setDongName] = useState("");
  const [industryName, setIndustryName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showResults, setShowResults] = useState(false);

  const [prediction, setPrediction] = useState(null);
  const [topIndustries, setTopIndustries] = useState([]);
  const [topRegions, setTopRegions] = useState([]);
  const [shapInsight, setShapInsight] = useState(null);
  const [aiReport, setAiReport] = useState(null);
  const [openScoreInfo, setOpenScoreInfo] = useState(false);
  const [activeInsight, setActiveInsight] = useState("strength"); // 강점/약점 탭 상태

  const WEIGHTS = {
    점포당_매출_금액: 0.35,
    안정성_지수: 0.25,
    성장성_지수: 0.20,
    입지_우위_지수: 0.20,
  };

  const selectedDong = useMemo(
    () => DONG_OPTIONS.find((d) => d.name === dongName) || null,
    [dongName]
  );
  const selectedIndustry = useMemo(
    () => INDUSTRY_OPTIONS.find((i) => i.name === industryName) || null,
    [industryName]
  );

  const resetAll = () => {
    setDongName("");
    setIndustryName("");
    setError("");
    setShowResults(false);
    setPrediction(null);
    setTopIndustries([]);
    setTopRegions([]);
    setShapInsight(null);
    setAiReport(null);
    setActiveInsight("strength");
  };

  const onAnalyze = async (e) => {
    e.preventDefault();
    setError("");
    setShapInsight(null);
    setAiReport(null);

    if (!selectedDong || !selectedIndustry) {
      setError("지역과 업종을 모두 선택(검색)해 주세요.");
      return;
    }

    setLoading(true);
    try {
      const params = new URLSearchParams({
        industry_code: selectedIndustry.code,
        dong_code: selectedDong.code,
      });

      // 5개의 API를 동시에 병렬로 호출하여 성능 최적화
      const responses = await Promise.all([
        fetch(`${API_BASE}/predict_by_selection`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dong_code: selectedDong.code, industry_code: selectedIndustry.code }),
        }),
        fetch(`${API_BASE}/recommend/industries?dong_code=${selectedDong.code}`),
        fetch(`${API_BASE}/recommend/regions?industry_code=${selectedIndustry.code}`),
        fetch(`${API_BASE}/get_insight?${params.toString()}`),
        fetch(`${API_BASE}/ai_insight?${params.toString()}`),
      ]);

      // 모든 응답이 정상적인지 확인
      for (const res of responses) {
        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: "알 수 없는 서버 오류" }));
          throw new Error(errData.detail || `${res.status} ${res.statusText}`);
        }
      }

      // 모든 응답 데이터를 JSON으로 파싱
      const [predData, indData, regData, shapData, aiData] = await Promise.all(
        responses.map(res => res.json())
      );

      // 모든 상태를 한 번에 업데이트
      setPrediction(predData);
      setTopIndustries(indData);
      setTopRegions(regData);
      setShapInsight(shapData);
      setAiReport(aiData);

      setShowResults(true);

    } catch (err) {
      setError(err.message || "분석 중 오류가 발생했습니다.");
      setShowResults(false);
    } finally {
      setLoading(false);
    }
  };

  const cbs = Math.round(Number(prediction?.cbs_score || 0));
  const donutStyle = {
    background: `conic-gradient(${MAIN_PURPLE} ${cbs * 3.6}deg, #EEEEEE ${cbs * 3.6}deg)`,
  };

  return (
    <div className="app-root">
      {!showResults ? (
        <div className="landing-wrap">
          <div className="landing-card">
            <img src="/logo.png" alt="Logo" className="logo-image" />
            <div className="logo-box">장사잘될지도</div>
            <p className="subtitle">AI 상권 분석 기반 창업 성공률 예측 서비스</p>
            <hr className="divider" />
            <form className="form" onSubmit={onAnalyze}>
              <div className="field-row">
                <label className="field-label">지역</label>
                <input
                  className="input"
                  placeholder="예: 역삼1동"
                  value={dongName}
                  onChange={(e) => setDongName(e.target.value)}
                  list="dong-list"
                />
                <datalist id="dong-list">
                  {DONG_OPTIONS.map((d) => (<option key={d.code} value={d.name} />))}
                </datalist>
              </div>
              <div className="field-row">
                <label className="field-label">업종</label>
                <input
                  className="input"
                  placeholder="예: 한식음식점"
                  value={industryName}
                  onChange={(e) => setIndustryName(e.target.value)}
                  list="industry-list"
                />
                <datalist id="industry-list">
                  {INDUSTRY_OPTIONS.map((i) => (<option key={i.code} value={i.name} />))}
                </datalist>
              </div>
              {error && <p className="error-text">{error}</p>}
              <button className="primary-btn" type="submit" disabled={loading}>
                {loading ? "분석 중..." : "성공률 분석"}
              </button>
            </form>
          </div>
        </div>
      ) : (
        <div className="dashboard">
          {/* ... 헤더 부분 (변경 없음) ... */}
          <div className="dash-header">
            <h1>장사잘될지도</h1>
            <div className="actions">
              <div className="action-buttons">
                <button className="ghost-btn" onClick={resetAll}>다시 입력하기</button>
                <button className="ghost-btn" onClick={() => window.print()}>PDF로 받기</button>
              </div>
              <p className="selection-note">
                선택하신 지역은 <b>{dongName}</b>, 업종은 <b>{industryName}</b> 입니다.
              </p>
            </div>
          </div>

          {/* 1행: 점수 + 매출 예측 (변경 없음) */}
          <div className="grid two equal-height">
            <div className="card donut-card">
              <h3 className="donut-title">종합 점수</h3>
              <div className="donut" style={donutStyle}>
                <div className="donut-hole"><span>{cbs} 점</span></div>
              </div>
              <div className="score-info">
                 <div className="score-brief">
                  <button type="button" className="info-link" onClick={() => setOpenScoreInfo(v => !v)} aria-expanded={openScoreInfo}>ⓘ 자세히</button>
                 </div>
                 {openScoreInfo && (
                   <div className="score-popover" role="dialog" aria-label="점수 산정 기준">
                    <div className="score-popover-head">
                      <b>점수 산정 기준</b>
                      <button className="score-popover-close" onClick={() => setOpenScoreInfo(false)}>✕</button>
                    </div>
                    <p className="formula">
                      점수 = (매출 × <b>{WEIGHTS.점포당_매출_금액*100}%</b>) + (안정성 × <b>{WEIGHTS.안정성_지수*100}%</b>) + (성장성 × <b>{WEIGHTS.성장성_지수*100}%</b>) + (입지 × <b>{WEIGHTS.입지_우위_지수*100}%</b>)
                    </p>
                   </div>
                 )}
               </div>
            </div>
            <div className="card">
              <h3>월 매출 예측</h3>
              <p className="revenue">{prediction ? Number(prediction.prediction).toLocaleString() : "–"} 원</p>
            </div>
          </div>

          {/* 2행: 강점/약점 + 최종 전략 */}
          <section className="insight-panel" aria-label="강점/약점 및 최종 전략">
            <div className="insight-header">
              <button type="button" className={`chip ${activeInsight === "strength" ? "active" : ""}`} onClick={() => setActiveInsight("strength")} aria-pressed={activeInsight === "strength"}>
                <span className="chip-dot" /> 강점 요소
              </button>
              <button type="button" className={`chip ${activeInsight === "weakness" ? "active" : ""}`} onClick={() => setActiveInsight("weakness")} aria-pressed={activeInsight === "weakness"}>
                <span className="chip-dot" /> 약점 요소
              </button>
            </div>

            <div className="insight-body">
              {/* shapInsight 데이터가 있을 때만 렌더링하도록 수정 */}
              {shapInsight ? (
                <ul className="bullet-list">
                  {(activeInsight === 'strength' ? shapInsight.strengths : shapInsight.weaknesses)
                    .map((item, index) => (
                      <li key={index}>{item.trim()}</li>
                    ))
                  }
                </ul>
              ) : (
                <p className="muted">분석 데이터를 불러오는 중입니다...</p>
              )}
            </div>

            <hr className="insight-divider" />

            <div className="strategy-block">
              <div className="strategy-heading">AI 컨설턴트의 최종 전략</div>
              <div className="strategy-content">
                 {/* aiReport 데이터가 있을 때만 렌더링하도록 수정 */}
                {aiReport ? (
                  <ParsedText text={aiReport.report} />
                ) : (
                  <p className="muted">AI 리포트를 생성 중입니다...</p>
                )}
              </div>
            </div>
          </section>

          {/* ... 3행, 4행 (변경 없음) ... */}
          <div className="grid two">
            <div className="card">
              <h3>{dongName}에서 성공확률 높은 업종 TOP 5</h3>
              <table className="table">
                <thead><tr><th>업종</th><th>점수</th></tr></thead>
                <tbody>
                  {topIndustries?.length ? (topIndustries.map((it, idx) => (
                    <tr key={idx}><td>{it.name}</td><td>{Number(it.cbs_score).toFixed(1)}</td></tr>
                  ))) : (<tr><td colSpan="2">추천 업종이 없습니다.</td></tr>)}
                </tbody>
              </table>
            </div>
            <div className="card">
              <h3>{industryName}으로 성공확률 높은 지역 TOP 5</h3>
              <table className="table">
                <thead><tr><th>지역</th><th>점수</th></tr></thead>
                <tbody>
                  {topRegions?.length ? (topRegions.map((it, idx) => (
                    <tr key={idx}><td>{it.name}</td><td>{Number(it.cbs_score).toFixed(1)}</td></tr>
                  ))) : (<tr><td colSpan="2">추천 지역이 없습니다.</td></tr>)}
                </tbody>
              </table>
            </div>
          </div>
          <div className="map-row">
            <div className="map-placeholder card"><div className="map-box">지도</div></div>
            <div className="list-card card">
              <h4>{dongName} · {industryName} 리스트</h4>
              <p className="muted">※ 지도/리스트는 추후 연동 예정</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
