// src/App.jsx
import React, { useEffect, useMemo, useState } from "react";
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

export default function App() {
  const [dongName, setDongName] = useState("");
  const [industryName, setIndustryName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showResults, setShowResults] = useState(false);

  const [prediction, setPrediction] = useState(null);
  const [topIndustries, setTopIndustries] = useState([]);
  const [topRegions, setTopRegions] = useState([]);

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
  };

  const onAnalyze = async (e) => {
    e.preventDefault();
    setError("");

    if (!selectedDong || !selectedIndustry) {
      setError("지역과 업종을 모두 선택(검색)해 주세요.");
      return;
    }

    setLoading(true);
    try {
      const resp1 = await fetch(`${API_BASE}/predict_by_selection`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dong_code: selectedDong.code,
          industry_code: selectedIndustry.code,
        }),
      });
      if (!resp1.ok) {
        const d = await resp1.json().catch(() => ({}));
        throw new Error(d.detail || "상권 분석 실패");
      }
      const pred = await resp1.json();
      setPrediction(pred);

      const resp2 = await fetch(`${API_BASE}/recommend/industries?dong_code=${selectedDong.code}`);
      setTopIndustries(await resp2.json());

      const resp3 = await fetch(`${API_BASE}/recommend/regions?industry_code=${selectedIndustry.code}`);
      setTopRegions(await resp3.json());

      setShowResults(true);
    } catch (err) {
      setError(err.message || "분석 중 오류가 발생했습니다.");
      setShowResults(false);
    } finally {
      setLoading(false);
    }
  };

  const cbs = Math.round(Number(prediction?.cbs_score || 0));
  const donutStyle = { background: `conic-gradient(#8A60E6 ${cbs * 3.6}deg, #EEEEEE ${cbs * 3.6}deg)` };

  return (
    <div className="app-root">
      {!showResults ? (
        <div className="landing-wrap">
          <div className="landing-card">
            <div className="logo-box">장사잘될지도 로고</div>
            <p className="subtitle">가나다라마바사아자차카타파하 서비스입니다.</p>
            <hr className="divider" />
            <form className="form" onSubmit={onAnalyze}>
              <div className="field-row">
                <label className="field-label">지역</label>
                <input
                  className="input"
                  placeholder="희망 지역을 선택하세요."
                  value={dongName}
                  onChange={(e) => setDongName(e.target.value)}
                  list="dong-list"
                />
                <datalist id="dong-list">
                  {DONG_OPTIONS.map((d) => (
                    <option key={d.code} value={d.name} />
                  ))}
                </datalist>
              </div>

              <div className="field-row">
                <label className="field-label">업종</label>
                <input
                  className="input"
                  placeholder="희망 업종을 선택하세요."
                  value={industryName}
                  onChange={(e) => setIndustryName(e.target.value)}
                  list="industry-list"
                />
                <datalist id="industry-list">
                  {INDUSTRY_OPTIONS.map((i) => (
                    <option key={i.code} value={i.name} />
                  ))}
                </datalist>
              </div>

              {error && <p className="error-text">{error}</p>}
              <button className="primary-btn" disabled={loading}>
                {loading ? "분석 중..." : "성공률 분석"}
              </button>
            </form>
          </div>
        </div>
      ) : (
        <div className="dashboard">
          <div className="dash-header">
            <h1>장사잘될지도</h1>
            <div className="actions">
              <div className="action-buttons">
                <button className="ghost-btn">PDF로 받기</button>
                <button className="ghost-btn" onClick={resetAll}>다시 입력하기</button>
              </div>
              <p className="selection-note">
                선택하신 지역은 <b>{dongName}</b>, 업종은 <b>{industryName}</b> 입니다.
              </p>
            </div>
          </div>

          {/* === 1행: 도넛 + 강점 === */}
          <div className="grid two equal-height">
            <div className="card donut-card">
              <h3>점수</h3>
              <div className="donut" style={donutStyle}>
                <div className="donut-hole"><span>{cbs} 점</span></div>
              </div>
            </div>

            <div className="card">
              <h3>강점 요소</h3>
              <ul className="bullet-list">
                {/* 내일 백엔드 논의 후 연결 예정 */}
                <li className="muted">강점 데이터는 추후 추가 예정</li>
              </ul>
            </div>
          </div>

          {/* === 2행: 예상 매출 + 약점 === */}
          <div className="grid two equal-height">
            <div className="card">
              <h3>예상 매출 금액</h3>
              <p className="revenue">
                {prediction ? Number(prediction.prediction).toLocaleString() : "–"} 원
              </p>
            </div>

            <div className="card">
              <h3>약점 요소</h3>
              <ul className="bullet-list">
                {/* 내일 백엔드 논의 후 연결 예정 */}
                <li className="muted">약점 데이터는 추후 추가 예정</li>
              </ul>
            </div>
          </div>

          {/* === 3행: TOP5 업종 / 지역 === */}
          <div className="grid two">
            <div className="card">
              <h3>{dongName}에서 성공확률 높은 업종 TOP 5</h3>
              <table className="table">
                <thead>
                  <tr>
                    <th>업종</th>
                    <th>점포 수</th>
                    <th>점수</th>
                  </tr>
                </thead>
                <tbody>
                  {topIndustries?.length ? (
                    topIndustries.map((it, idx) => (
                      <tr key={idx}>
                        <td>{it.name}</td>
                        <td>{it.store_count ?? "-"}</td>
                        <td>{Number(it.cbs_score).toFixed(1)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr><td colSpan="3">추천 업종이 없습니다.</td></tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="card">
              <h3>{industryName}으로 성공확률 높은 지역 TOP 5</h3>
              <table className="table">
                <thead>
                  <tr>
                    <th>지역</th>
                    <th>점포 수</th>
                    <th>점수</th>
                  </tr>
                </thead>
                <tbody>
                  {topRegions?.length ? (
                    topRegions.map((it, idx) => (
                      <tr key={idx}>
                        <td>{it.name}</td>
                        <td>{it.store_count ?? "-"}</td>
                        <td>{Number(it.cbs_score).toFixed(1)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr><td colSpan="3">추천 지역이 없습니다.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* === 4행: 지도 + 리스트 === */}
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
