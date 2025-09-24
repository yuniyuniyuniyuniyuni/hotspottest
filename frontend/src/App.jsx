import React, { useState } from "react";
import "./App.css";

function App() {
  const [region, setRegion] = useState("성수동");
  const [category, setCategory] = useState("카페");
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    const res = await fetch(
      `http://localhost:8000/predict?region=${region}&category=${category}`
    );
    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="App">
      <h1>장사잘될지도 (MVP)</h1>
      <div className="input-box">
        <input
          type="text"
          value={region}
          onChange={(e) => setRegion(e.target.value)}
          placeholder="지역 입력"
        />
        <input
          type="text"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          placeholder="업종 입력"
        />
        <button onClick={handlePredict}>성공률 예측</button>
      </div>

      {result && (
        <div className="result-box">
          <h3>결과</h3>
          <p><strong>지역:</strong> {result.region}</p>
          <p><strong>업종:</strong> {result.category}</p>
          <p><strong>예상 성공률:</strong> {result.success_rate}</p>
          <p><strong>위험 요인:</strong> {result.risk_factors.join(", ")}</p>
          <p><strong>성공 요인:</strong> {result.success_factors.join(", ")}</p>
        </div>
      )}
    </div>
  );
}

export default App;
