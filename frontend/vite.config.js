import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // ✅ 추가된 부분: 로컬 개발 서버 설정을 추가합니다.
  server: {
    proxy: {
      // '/api'로 시작하는 모든 요청을 백엔드 서버(localhost:8000)로 전달합니다.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    }
  }
})