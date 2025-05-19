<template>
  <div class="dashboard-container">
    <main class="main-content">
      <!-- 상단 로고 / 설명 -->
      <section class="main-header header-section">
        <div>
          <h2 class="logo-title">AISM</h2>
          <p class="subtitle">AI 서비스 운영에서 발생하는 LLM 등 모델 사용량을 분석하고, 과도한 소비 원인을 추적하여 비용과 네트워크 트래픽을 최적화하는 모니터링 솔루션입니다.</p>
        </div>
      </section>

      <!-- 기능 카드 -->
      <section class="card-row">
        <div class="card-item">
          <span>Upload</span>
          <label for="file-upload" class="circle-btn">
            <i class="bi bi-upload"></i>
          </label>
          <input id="file-upload" type="file" accept=".csv" @change="handleFileUpload" style="display: none;" />
        </div>
        <div class="card-item">
          <span>Download</span>
          <button class="circle-btn"><i class="bi bi-download" @click="downloadGraph"></i></button>
        </div>
        <div class="card-item">
          <span>Token Count</span>
          <span style="margin-left: 8px; font-weight: bold;">{{ Math.round(tokenCount) }}</span>
          <button class="circle-btn">
            <i class="bi bi-123"></i>
          </button>
        </div>
      </section>


      <!-- 하단 그래프 -->
      <!-- detailed graph -->
      <section class="graph-row">
        <ul class="list-group">
          <li class="list-group-item" @click="selectDataType('generation')">Generation</li>
          <li class="list-group-item" @click="selectDataType('embedding')">Embedding</li>
          <li class="list-group-item" @click="selectDataType('summarization')">Summarization</li>
        </ul>
        <div class="graph-box">
          <!-- <h5>Detailed Graph</h5> -->
          <img v-if="selectedType === 'generation' && generationGraph" :src="generationGraph"
            alt="Generation Graph" class="result-image" />

          <img v-else-if="selectedType === 'embedding' && embeddingGraph" :src="embeddingGraph"
            alt="Embedding Graph" class="result-image" />

          <img v-else-if="selectedType === 'summarization' && summarizationGraph"
            :src="summarizationGraph" alt="Summarization Graph" class="result-image" />

          <p v-else style="color: #aaa;">그래프를 불러올 수 없습니다.</p>
        </div>
        <!-- overall graph -->
        <div class="graph-box">
          <!-- <h5>Overall Graph</h5> -->
          <img v-if="overallGraph" :src="overallGraph" alt="Overall Graph" class="result-image" />
          <!-- <button @click="downloadImage(results.overall_graph)" class="download-btn">Download</button> -->
           <p v-else style="color: #aaa;">그래프를 불러올 수 없습니다.</p>
        </div>
      </section>
    </main>
  </div>
</template>



<script setup>
import { ref, onMounted } from 'vue'

const tokenCount = ref(0)
const requestCount = ref(0)
const selectedType = ref('generation')  // 기본값: generation

const savedFilename = ref('')

const generationGraph = ref('')
const embeddingGraph = ref('')
const summarizationGraph = ref('')
const overallGraph = ref('')

//1. 업로드
async function handleFileUpload(event) {
  const file = event.target.files[0]
  if (!file) return

  const formData = new FormData()
  formData.append('file', file)

  try {
    alert('업로드가 요청되었습니다.')

    const response = await fetch('http://localhost:8001/upload', {
      method: 'POST',
      body: formData,
    })

    const data = await response.json()

    // FastAPI 응답값 반영
    tokenCount.value = data.token_count
    requestCount.value = data.request_count
    savedFilename.value = data.saved_filename

    generationGraph.value = data.generation_graph
    embeddingGraph.value = data.embedding_graph
    summarizationGraph.value = data.summarization_graph
    overallGraph.value = data.overall_graph

    selectedType.value = 'generation' // 업로드 후 기본 그래프 설정

    alert('업로드가 정상적으로 완료되었습니다.')
  } catch (err) {
    console.error('업로드 실패:', err)
  }
}

function downloadGraph() {
  // 예: 현재 선택된 타입(generation, embedding 등)을 기준으로 그래프 선택
  let graphPath = ''
  if (selectedType.value === 'generation') {
    graphPath = generationGraph.value
  } else if (selectedType.value === 'embedding') {
    graphPath = embeddingGraph.value
  } else if (selectedType.value === 'summarization') {
    graphPath = summarizationGraph.value
  } else if (selectedType.value === 'overall') {
    graphPath = overallGraph.value
  }

  // 경로가 비어있으면 중단
  if (!graphPath) {
    console.error('그래프 경로가 없습니다.')
    return
  }

  // fetch로 이미지 blob 받아서 다운로드 처리
  fetch(graphPath)
    .then(res => res.blob())
    .then(blob => {
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'graph.png'
      document.body.appendChild(link)
      link.click()
      link.remove()
    })
    .catch(err => {
      console.error('그래프 다운로드 실패:', err)
    })
}

//4. Graph
function selectDataType(type) {
  selectedType.value = type
}

</script>

<style>
body {
  margin: 0;
  background-color: #1e1e1e !important;
  color: white;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
</style>

<style scoped>
.header-section {
  padding: 1rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  /* 구분선 느낌 */
  text-align: left;
}


.dashboard-container {
  background-color: #1e1e1e;
  min-height: 100vh;
  color: #ffffff;
  padding: 2rem;
  box-sizing: border-box;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.box {
  background-color: #2a2a2a;
  padding: 2rem;
  border-radius: 1rem;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.6);
}

.logo-title {
  font-size: 2rem;
  color: #f0c293;
  font-weight: 700;
  margin-bottom: 0.5rem;
}

.subtitle {
  color: #cccccc;
  font-size: 1rem;
  line-height: 1.6;
}

.card-row {
  display: flex;
  flex-wrap: nowrap;
  gap: 1.5rem;
  justify-content: space-between;
}

.card-item {
  flex: 1 1 calc(25% - 1rem);
  background-color: #2f2f2f;
  border: 1px solid rgba(240, 194, 147, 0.3);
  color: #f0c293;
  padding: 1.5rem;
  border-radius: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  transition: transform 0.2s ease;
}

.card-item:hover {
  transform: translateY(-4px);
}

.circle-btn {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: linear-gradient(135deg, #f0c293, #f5d3aa);
  border: none;
  color: #1e1e1e;
  font-weight: bold;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
  transition: transform 0.1s ease-in-out;

  /* 아이콘 정확한 가운데 정렬 */
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
}

.circle-btn:hover {
  transform: scale(1.1);
}

.graph-row {
  display: flex;
  flex-wrap: nowrap;
  gap: 1.5rem;
}

.graph-box {
  flex: 1 1 48%;
  background-color: #2a2a2a;
  padding: 2rem;
  border-radius: 1rem;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
  color: #ffffff;
  text-align: center;
  height: auto;
  min-height: 300px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.graph-box img {
  max-width: 100%; /* 이미지 최대 너비는 부모 요소에 맞춤 */
  width: 100%; /* 이미지가 박스 너비에 꽉 차게 설정 */
  height: auto; /* 높이는 원본 비율에 맞게 자동 조정 */
  object-fit: contain; /* 이미지 비율 유지 */
}

.graph-box h5 {
  font-size: 1.25rem;
  color: #f0c293;
}

.list-group {
  list-style: none;
  margin: 0;
  padding: 2rem 1.5rem;
  background-color: #2a2a2a;
  border-radius: 1rem;
  min-height: 300px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  gap: 0.75rem;
  flex: 1 1 20%;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
}

.list-group-item {
  padding: 0.5rem 0.5rem;
  color: #f0c293;
  font-size: 0.95rem;
  border: none;
  background: transparent;
  border-radius: 0.4rem;
  cursor: default;
  transition: background-color 0.2s ease;
}

.list-group-item:hover {
  background-color: rgba(255, 255, 255, 0.05);
}
</style>