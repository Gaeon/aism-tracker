# 🧠 AISM Tracker (AI Service Management)

AI 서비스를 운용하며 발생하는 **토큰 사용량과 호출 트래픽**을 예측하고, **비효율적인 사용 구간을 분석**하여 자동 대응하는 **AIOps 기반 관리 시스템**입니다.

---

## 📌 프로젝트 개요

AISM Tracker는 다음을 목적으로 합니다:

- LLM 호출 시 사용되는 **토큰 수 예측**
- 과도한 사용 패턴의 **원인 분석 및 최적화 제안**
- 예측 결과 시각화 및 **비용 전망 리포트 제공**
- 예측 오차(RMSE)가 높을 경우 **모델 자동 재학습**

---

## 🧩 폴더 구조
```
├── public/                    # Vite 기반 프론트엔드
│   ├── index.html
│   ├── node_modules/
│   ├── package-lock.json
│   ├── package.json
│   ├── src/                  # 프론트엔드 코드
│   └── vite.config.js
├── readme.md
├── server/
│   ├── model/                # 저장된 학습 모델
│   ├── model-images/         # 시각화 이미지 저장
│   ├── uploaded_files/       # 업로드된 시계열 로그
│   └── view-model-architecture/ # 모델 구조 이미지 등
└── server_model/
├── AISM_LSTM.ipynb       # 모델 학습/테스트 노트북
├── config.py             # 설정 파일
├── main.py               # FastAPI 엔트리포인트
├── model.py              # LSTM 모델 구조 정의
└── weight_used_model.py  # 예측/재학습 로직
```

---

## ⚙️ 기술 스택

| 구성 요소        | 기술                |
|------------------|---------------------|
| 프론트엔드        | Vite + Vue.js (예정) |
| 백엔드 API 서버   | FastAPI             |
| 예측 모델        | LSTM (PyTorch 또는 Keras 기반) |
| 시각화            | matplotlib 등       |
| 재학습 조건       | RMSE 기준 초과 시 자동 수행 |

---

## 🚀 실행 방법

### 1️⃣ 백엔드 실행 (FastAPI)

```bash

# FastAPI 실행
uvicorn server_model.main:app --reload --port 8001
```

### 2️⃣ 프론트엔드 실행 (Vite)

```bash
cd public

# 모듈 설치
npm install

# 개발 서버 실행
npm run dev
```
---

## 🧠 주요 기능
	- ✅ 토큰 사용량 예측 (LSTM 기반)
	- ✅ 실제 vs 예측 비교 그래프 출력
	- ✅ 비정상 사용 패턴 탐지 및 수정 제안
	- ✅ 자동 재학습 (RMSE 기준)
	- ✅ API 시각화 및 예측 리포트 API 제공
 
---

### 🙋 팀 소개

1기 2반 7조
김가언, 김재현, 서찬영, 유소영, 이현희, 최혜정

---
### 기획서

[AISM Tracker - 7조.pdf](https://github.com/user-attachments/files/20280789/AISM.Tracker.-.7.pdf)
