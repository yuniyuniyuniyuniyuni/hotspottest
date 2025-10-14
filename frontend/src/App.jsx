// src/App.jsx
import React, { useState, useCallback, useEffect } from "react";
import { APIProvider, useMapsLibrary } from "@vis.gl/react-google-maps"; 
import "./App.css";
import DONG_OPTIONS from "./data/dongOptions";
import MapComponent from "./components/MapComponent"; 

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

const WEIGHTS = {
  "점포당_매출_금액": "35%",
  "안정성_지수": "25%",
  "성장성_지수": "20%",
  "입지_우위_지수": "20%",
};

const API_BASE = import.meta.env?.VITE_API_BASE || "http://localhost:8000";
const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
const MAIN_PURPLE = "#8A60E6";

// AI 응답 텍스트의 줄바꿈을 처리하기 위한 컴포넌트
const ParsedText = ({ text }) => {
  if (typeof text !== 'string' || text.trim() === "") {
    return null;
  }
  return text.split('\n').map((paragraph, index) => (
    <p key={index} className="strategy-paragraph">
      {paragraph || <br />}
    </p>
  ));
};

function MapAndListComponent({ dongName, industryName }) {
  const [places, setPlaces] = useState([]);
  const [center, setCenter] = useState({ lat: 37.5665, lng: 126.9780 });

  const placesLibrary = useMapsLibrary('places');
  
  useEffect(() => {
    if (!placesLibrary || !dongName || !industryName) return;

    const { Place } = placesLibrary;
    const { LatLngBounds } = window.google.maps;

    const searchPlaces = async () => {
      const request = {
        textQuery: `${dongName} ${industryName}`,
        fields: ["id", "displayName", "location", "formattedAddress"],
        language: 'ko',
      };
      try {
        const { places: searchResults } = await Place.searchByText(request);
        if (searchResults.length > 0) {
          setPlaces(searchResults);
          const bounds = new LatLngBounds();
          searchResults.forEach(p => p.location && bounds.extend(p.location));
          setCenter(bounds.getCenter().toJSON());
        } else {
          setPlaces([]);
        }
      } catch (error) {
        console.error("Places API 검색 실패:", error);
      }
    };
    searchPlaces();
  }, [placesLibrary, dongName, industryName]);

  return (
    <div className="map-row">
      {/* MapComponent에는 이제 props로 데이터만 전달합니다. */}
      <MapComponent 
        places={places} 
        center={center} 
        dongName={dongName} 
        industryName={industryName} 
      />
      
      {/* ★★★ 기존의 리스트 플레이스홀더를 실제 데이터 리스트로 교체합니다. ★★★ */}
      <div className="list-card card">
        <h4>{dongName} · {industryName} 검색 결과</h4>
        {places.length > 0 ? (
          <ul className="places-list">
            {places.map(place => (
              <li key={place.id}>
                <strong>{place.displayName}</strong>
                <p>{place.formattedAddress}</p>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">검색된 장소가 없습니다.</p>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [dongCode, setDongCode] = useState("");
  const [industryCode, setIndustryCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [avgScores, setAvgScores] = useState(null); 

  const [prediction, setPrediction] = useState(null);
  const [topIndustries, setTopIndustries] = useState([]);
  const [topRegions, setTopRegions] = useState([]);
  const [shapInsight, setShapInsight] = useState(null);
  const [aiReport, setAiReport] = useState(null);
  const [openScoreInfo, setOpenScoreInfo] = useState(false);
  const [activeInsight, setActiveInsight] = useState("strength");

  const dongName = DONG_OPTIONS.find(d => d.code === dongCode)?.name || "";
  const industryName = INDUSTRY_OPTIONS.find(i => i.code === industryCode)?.name || "";

  const resetAll = () => {
    setDongCode("");
    setIndustryCode("");
    setError("");
    setShowResults(false);
    setPrediction(null);
    setTopIndustries([]);
    setTopRegions([]);
    setShapInsight(null);
    setAiReport(null);
    setActiveInsight("strength");
  };

  // ★★★ 분석 로직을 재사용 가능한 함수로 분리 ★★★
  const startAnalysis = useCallback(async (dCode, iCode) => {
    if (!dCode || !iCode) {
      setError("지역과 업종 코드가 올바르지 않습니다.");
      return;
    }
    
    setError("");
    setLoading(true);
    setShapInsight(null); // 분석 시작 시 이전 인사이트 초기화
    setAiReport(null);    // 분석 시작 시 이전 리포트 초기화
    setAvgScores(null);

    // 상태 업데이트로 화면에 선택된 지역/업종 이름 즉시 반영
    setDongCode(dCode);
    setIndustryCode(iCode);

    try {
      const params = new URLSearchParams({ industry_code: iCode, dong_code: dCode });
      
      const responses = await Promise.all([
        fetch(`${API_BASE}/predict_by_selection`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dong_code: dCode, industry_code: iCode }),
        }),
        fetch(`${API_BASE}/recommend/industries?dong_code=${dCode}`),
        fetch(`${API_BASE}/recommend/regions?industry_code=${iCode}`),
        fetch(`${API_BASE}/get_insight?${params.toString()}`),
        fetch(`${API_BASE}/ai_insight?${params.toString()}`),
        fetch(`${API_BASE}/stats?${params.toString()}`),
      ]);

      for (const res of responses) {
        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: "알 수 없는 서버 오류" }));
          throw new Error(errData.detail || `${res.status} ${res.statusText}`);
        }
      }
      
      const [predData, indData, regData, shapData, aiData, statsData] = await Promise.all(
        responses.map(res => res.json())
      );
      
      setPrediction(predData);
      setTopIndustries(indData);
      setTopRegions(regData);
      setShapInsight(shapData);
      setAiReport(aiData);
      setAvgScores(statsData);
      setShowResults(true);

    } catch (err) {
      setError(err.message || "분석 중 오류가 발생했습니다.");
      setShowResults(false);
    } finally {
      setLoading(false);
    }
  }, []); // 종속성 배열이 비어있으므로 함수는 한 번만 생성됨

  const handleSubmit = (e) => {
    e.preventDefault();
    const selectedDong = DONG_OPTIONS.find(d => d.name === e.target.elements.dong.value);
    const selectedIndustry = INDUSTRY_OPTIONS.find(i => i.name === e.target.elements.industry.value);
    
    if (!selectedDong || !selectedIndustry) {
      setError("지역과 업종을 모두 선택(검색)해 주세요.");
      return;
    }
    startAnalysis(selectedDong.code, selectedIndustry.code);
  };

  const cbs = Number(prediction?.cbs_score || 0).toFixed(1);
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
            <span className="subtitle">데이터 기반 창업 성공 내비게이션</span>
            <p className="subtitle">
            창업자가 자신의 조건(업종, 지역)을 입력하면, 인공지능 기반 데이터 분석을 통해 예상 매출을 산출하고
            이를 기반으로 점수화, 강·약점 분석, 최적의 대안을 제공하는 서비스입니다.
            </p>
            <hr className="divider" />
            <form className="form" onSubmit={handleSubmit}>
              <div className="field-row">
                <label className="field-label">지역</label>
                <input name="dong" className="input" placeholder="예: 역삼1동" list="dong-list" defaultValue={dongName} />
                <datalist id="dong-list">
                  {DONG_OPTIONS.map((d) => (<option key={d.code} value={d.name} />))}
                </datalist>
              </div>
              <div className="field-row">
                <label className="field-label">업종</label>
                <input name="industry" className="input" placeholder="예: 한식음식점" list="industry-list" defaultValue={industryName} />
                <datalist id="industry-list">
                  {INDUSTRY_OPTIONS.map((i) => (<option key={i.code} value={i.name} />))}
                </datalist>
              </div>
              {error && <p className="error-text">{error}</p>}
              <button className="primary-btn" type="submit" disabled={loading}>
                {loading && <span className="btn-spinner" aria-hidden="true" />}
                {loading ? "분석 중..." : "성공률 분석"}
              </button>
            </form>
          </div>
        </div>
      ) : (
        <APIProvider apiKey={GOOGLE_MAPS_API_KEY} libraries={['places', 'maps']}>
          <div className="dashboard">
            <div className="dash-header">
              <h1>장사잘될지도</h1>
              <p className="selection-note">
                  선택하신 지역은 <b>{dongName}</b>, 업종은 <b>{industryName}</b> 입니다.
                </p>
              <div className="actions">
                <div className="action-buttons">
                  <button className="ghost-btn" onClick={resetAll}>다시 입력하기</button>
                  <button className="ghost-btn" onClick={() => window.print()}>PDF로 받기</button>
                </div>
              </div>
            </div>

            <div className="grid two equal-height">
              <div className="card donut-card">
                <h3 className="donut-title">종합 점수</h3>
                <div className="donut" style={donutStyle}>
                  <div className="donut-hole"><span>{cbs} 점</span></div>
                </div>
                {avgScores && <p className="average-note">서울시 외식업군 전체 평균: {avgScores.avg_cbs_score_seoul.toLocaleString()}점</p>}
                <div className="score-info">
                  <button type="button" className="info-link" onClick={() => setOpenScoreInfo(v => !v)}>ⓘ 자세히</button>
                </div>
                {openScoreInfo && (
                  <div className="score-popover" role="dialog" aria-label="점수 산정 기준">
                    <div className="score-popover-head">
                      <b>점수 산정 기준</b>
                      <button
                        className="score-popover-close"
                        onClick={() => setOpenScoreInfo(false)}
                      >
                        ✕
                      </button>
                    </div>

                    <p className="formula">
                      CBS 점수 = (점포당 매출 금액 × <b>{WEIGHTS.점포당_매출_금액}</b>) +
                      (안정성 지수 × <b>{WEIGHTS.안정성_지수}</b>) + <br />
                      (성장성 지수 × <b>{WEIGHTS.성장성_지수}</b>) +
                      (입지 우위 지수 × <b>{WEIGHTS.입지_우위_지수}</b>)
                    </p>

                    <table className="table compact">
                      <thead>
                        <tr><th>요소</th><th>가중치</th><th>설명</th></tr>
                      </thead>
                      <tbody>
                        <tr><td>점포당 매출 금액</td><td>35%</td><td>동·업종 단위의 평균 매출</td></tr>
                        <tr><td>안정성 지수</td><td>25%</td><td>폐업률, 변동성 등 리스크</td></tr>
                        <tr><td>성장성 지수</td><td>20%</td><td>최근 3년간 매출 추세</td></tr>
                        <tr><td>입지 우위 지수</td><td>20%</td><td>유동·접근성·경쟁 밀도</td></tr>
                      </tbody>
                    </table>

                    <p className="muted small">
                      ※ 데이터는 주기적으로 업데이트되며, 가중치는 모델에 따라 조정될 수 있습니다.
                    </p>
                  </div>
                )}
                <hr className="insight-divider" />
                <div className="insight-header">
                  <button type="button" className="chip active">CBS 점수 결정 요인</button>
                </div>
                <div className="insight-body">
                  {shapInsight ? (<ul className="bullet-list">
                      {shapInsight.cbs.map((item, index) => (<li key={index}>{item.trim()}</li>))}
                  </ul>) : (<p className="muted">분석 데이터를 불러오는 중입니다...</p>)}
                </div>
              </div>
              <div className="card revenue-card">
                <div className="revenue-header">
                  <h3>월 매출 예측</h3>
                </div>
                <div className="revenue"><p>{prediction ? Number(prediction.prediction).toLocaleString() : "–"} 원</p></div>
                {avgScores && (
                  <div className="average-note-group">
                    <p className="average-note">
                      {dongName} 외식업군 전체 업종 평균: {avgScores.avg_sales_dong.toLocaleString()} 원
                    </p>
                    <p className="average-note">
                      {industryName} 서울시 전체 지역 평균: {avgScores.avg_sales_industry.toLocaleString()} 원
                    </p>
                  </div>
                )}

                <hr className="insight-divider" />
                <div className="insight-header">
                    <button type="button" className={`chip ${activeInsight === "strength" ? "active" : ""}`} onClick={() => setActiveInsight("strength")}>강점 요소</button>
                    <button type="button" className={`chip ${activeInsight === "weakness" ? "active" : ""}`} onClick={() => setActiveInsight("weakness")}>약점 요소</button>
                </div>
                <div className="insight-body">
                  {shapInsight ? (<ul className="bullet-list">
                      {(activeInsight === 'strength' ? shapInsight.strengths : shapInsight.weaknesses).map((item, index) => (
                        <li key={index}>{item.trim()}</li>
                      ))}
                  </ul>) : (<p className="muted">분석 데이터를 불러오는 중입니다...</p>)}
                </div>
              </div>
            </div>
            <div className="grid equal-height narrow-gap">
              <div className="card">
                <h3><b>{dongName}</b> 평균 임대료</h3>
                <div className="extra-card-body">
                  <p className="muted">000000원</p>
                </div>
              </div>
              <div className="card">
                <h3>평당 <b>{industryName}</b> 인테리어 비용</h3>
                <div className="extra-card-body">
                  <p className="muted">000000원</p>
                </div>
              </div>
            </div>
            {/* ★★★ AI 리포트 출력 형식 수정 ★★★ */}
            <section className="insight-panel" aria-label="AI 컨설턴트 최종 전략">
              <div className="strategy-block">
                <div className="strategy-heading">AI 컨설턴트의 최종 전략</div>
                <div className="strategy-content">
                  {aiReport && aiReport.report ? (
                    <>
                      <h4 className="ai-report-subtitle">결론 요약</h4>
                      <ParsedText text={aiReport.report.summary} />
                      <h4 className="ai-report-subtitle">CBS 결정 요인 분석</h4>
                      <ParsedText text={aiReport.report.cbs_analysis} />
                      <h4 className="ai-report-subtitle">강점 및 약점 평가</h4>
                      <ParsedText text={aiReport.report.evaluation} />
                      <h4 className="ai-report-subtitle">최종 전략 제언</h4>
                      <ParsedText text={aiReport.report.strategy} />
                    </>
                  ) : (
                    <p className="muted">AI 리포트를 생성 중입니다...</p>
                  )}
                </div>
              </div>
            </section>

            {/* ★★★ 추천 테이블에 '바로 분석' 버튼 추가 ★★★ */}
            <div className="grid two">
              <div className="card">
                <h3>{dongName}에서 성공확률 높은 업종 TOP 5</h3>
                <table className="table">
                  <thead><tr><th>업종</th><th>점수</th><th></th></tr></thead>
                  <tbody>
                    {topIndustries?.length ? (topIndustries.map((it) => (
                      <tr key={it.code}>
                        <td>{it.name}</td>
                        <td>{Number(it.cbs_score).toFixed(1)}</td>
                        <td>
                          <button className="table-action-btn" onClick={() => startAnalysis(dongCode, it.code)} disabled={loading}>
                            바로 분석
                          </button>
                        </td>
                      </tr>
                    ))) : (<tr><td colSpan="3">추천 업종이 없습니다.</td></tr>)}
                  </tbody>
                </table>
              </div>
              <div className="card">
                <h3>{industryName}으로 성공확률 높은 지역 TOP 5</h3>
                <table className="table">
                  <thead><tr><th>지역</th><th>점수</th><th></th></tr></thead>
                  <tbody>
                    {topRegions?.length ? (topRegions.map((it) => (
                      <tr key={it.code}>
                        <td>{it.name}</td>
                        <td>{Number(it.cbs_score).toFixed(1)}</td>
                        <td>
                          <button className="table-action-btn" onClick={() => startAnalysis(it.code, industryCode)} disabled={loading}>
                            바로 분석
                          </button>
                        </td>
                      </tr>
                    ))) : (<tr><td colSpan="3">추천 지역이 없습니다.</td></tr>)}
                  </tbody>
                </table>
              </div>
            </div>
            
            <MapAndListComponent dongName={dongName} industryName={industryName} />
          </div>
        </APIProvider>  
      )}
    </div>
  );
}
