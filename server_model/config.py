
import os

# 기본 경로 설정 (환경 변수에서 가져오거나 기본값 사용)
BASE_DIR = os.getenv("BASE_DIR", "../server")
# /Users/gaeon/workspace/SKALA/DataAnalysis/MLOps/model_serving_rpt/server

# 상대 경로를 연결할 때 슬래시(/) 제거
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_files")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMAGE_DIR = os.path.join(BASE_DIR, "view-model-architecture") 
MODEL_IMG_DIR = os.path.join(MODEL_DIR, "result") #os.path.join(BASE_DIR, "model-images")
MODEL_ARCHI_DIR = os.path.join(IMAGE_DIR, "shapes")

# 파일 경로 설정
DATA_PATH = os.path.join(UPLOAD_DIR, "2020_2025_Data.csv")
MODEL_SAVE_PATH = {'Generation': os.path.join(MODEL_DIR, "generation_lstm_model_nogpu.keras"),
                   'Summarization': os.path.join(MODEL_DIR, "summarization_lstm_model_nogpu.keras"),
                   'Embedding': os.path.join(MODEL_DIR, "embedding_lstm_model_nogpu.keras")
                   }
#MODEL_PLOT_PATH = os.path.join(IMAGE_DIR, "model.png")
MODEL_SHAPES_PLOT_PATH = os.path.join(MODEL_ARCHI_DIR, "model_shapes.png")
MODEL_ARCHI_PLOT_PATH = os.path.join(MODEL_ARCHI_DIR, "model_architecture.png")

PREDICTION_PLOT_PATH = {'Generation': os.path.join(MODEL_IMG_DIR, "generation_API_calls.png"),
                        'Summarization': os.path.join(MODEL_IMG_DIR, "summarization_API_calls.png"),
                        'Embedding': os.path.join(MODEL_IMG_DIR, "embedding_API_calls.png"),
                        'All token': os.path.join(MODEL_IMG_DIR, "AllToken_API_calls.png")
                        }

# 모델 관련 파라미터
THRESHOLD = {'Generation': 500, 
             'Summarization': 500,
             'Embedding': 500
            }
TIMESTAMP = 7 #30
PREDICT_START = '2025-02-01'
HORIZON = 7