import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx"; // App.jsx 경로를 명확하게 수정
import "./index.css"; // 이 부분의 경로를 수정합니다.

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

