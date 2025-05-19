import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      // '/upload': 'http://localhost:8000',
      '/download': 'http://localhost:8001',
      '/token_count': {target : 'http://localhost:8001', proxyTimeout: 5000},
      '/request_count': 'http://localhost:8001',
    }
  }
})
