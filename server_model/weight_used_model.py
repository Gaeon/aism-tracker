# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import load_model # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from keras.utils import plot_model # type: ignore
import os
from config import MODEL_SAVE_PATH, MODEL_ARCHI_PLOT_PATH, MODEL_SHAPES_PLOT_PATH, \
                   PREDICTION_PLOT_PATH, THRESHOLD, TIMESTAMP, PREDICT_START, HORIZON

# 모델 로딩
# model = load_model(MODEL_SAVE_PATH['Generation'])

# 데이터 로딩
#dataset = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=['Date'], encoding='utf-8')

# 모델 아키텍처 이미지 생성
# plot_model(model, to_file=MODEL_ARCHI_PLOT_PATH)
# plot_model(model, to_file=MODEL_SHAPES_PLOT_PATH, show_shapes=True)


# RMSE 계산 함수
def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    result_msg = f"The root mean squared error is {rmse}."
    print(result_msg)
    return rmse

# 예측 결과 그래프 저장 함수
def plot_predictions(test, predicted, col):
    plt.clf()  # 이전 그래프 초기화
    plt.plot(test, color='red', label='Real API Token Usage')
    plt.plot(predicted, color='blue', label='Predicted API Token Usages')
    plt.title('API Token Usage Prediction')
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend()
    plt.tight_layout() 
    plt.savefig(PREDICTION_PLOT_PATH[col])
    return PREDICTION_PLOT_PATH[col]


# 전처리 및 예측 진행 함수 + 임계치 달성 확인
def process_(dataset, col, threshold, timestamp=TIMESTAMP, scaler=False):
    model = load_model(MODEL_SAVE_PATH[col])

    if scaler is False:
        training_set = pd.DataFrame(dataset[:'2023'][col]).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        _ = scaler.fit_transform(training_set.reshape(-1, 1)).flatten()

    test_set = pd.DataFrame(dataset[PREDICT_START:][col]).values
    test_scaled = scaler.transform(test_set.reshape(-1, 1)).flatten()

    # 테스트용 윈도우 생성
    X_test = []
    for i in range(len(test_scaled) - TIMESTAMP):
        X_test.append(test_scaled[i : i + TIMESTAMP])

    X_test = np.array(X_test).reshape(-1, TIMESTAMP, 1)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    # rmse 확인
    result_rmse = return_rmse(test_set[TIMESTAMP:], y_pred)

    if result_rmse > threshold: # 재학습
        #retrain(dataset, col=col)
        return retrain_new(dataset, col=col)
    else:
        # rolling forcast
        last_window = X_test[-1]

        future_scaled = []

        x_input = last_window.reshape(-1, TIMESTAMP, 1) 
        for _ in range(HORIZON):
            next_scaled = model.predict(x_input)[0, 0]
            future_scaled.append(next_scaled)
            new_window = np.concatenate([x_input[0, 1:, :], [[next_scaled]]], axis=0)
            x_input = new_window.reshape(1, TIMESTAMP, 1)

        future_scaled = np.array(future_scaled).reshape(-1, 1)
        future = scaler.inverse_transform(future_scaled).flatten()
        recent_forcast = scaler.inverse_transform(future_scaled).flatten()[0]

        # visualization
        start_date = dataset[PREDICT_START:].index[TIMESTAMP] 
        end_date = dataset.index[-1] + pd.Timedelta(days=HORIZON)
        future_index = pd.date_range(start=start_date, end=end_date, freq='D', name='date')
        
        y_all = pd.Series(np.concatenate([y_pred, future]), index=future_index)

        result_visualizing = plot_predictions(dataset[start_date:][col], y_all, col)
        
        return result_rmse, result_visualizing, y_all, recent_forcast

# 모델 재학습 함수
def retrain(dataset, col, timestamp=TIMESTAMP, scaler=False):

    training_set = pd.DataFrame(dataset['2024-01-01':'2024-12-31'][col]).values

    # 학습 데이터 전체 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(training_set.reshape(-1, 1)).flatten()

    X_train, y_train = [], []
    for i in range(len(train_scaled) - TIMESTAMP):
        X_train.append(train_scaled[i : i + TIMESTAMP])
        y_train.append(train_scaled[i + TIMESTAMP])

    X_train = np.array(X_train).reshape(-1, TIMESTAMP, 1)
    y_train = np.array(y_train)

    # 모델 로드 후 재학습
    model = load_model(MODEL_SAVE_PATH[col])
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # 모델 다시 저장
    model.save(MODEL_SAVE_PATH[col])
    return process_(dataset, col=col, threshold=THRESHOLD[col], scaler=scaler)


# 모델 재학습(아예 가중치 모두 다시 학습) 함수
def retrain_new(dataset, col, timestamp=TIMESTAMP):

    training_set = pd.DataFrame(dataset['2024-01-01':'2024-12-31'][col]).values

    # 학습 데이터 전체 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(training_set.reshape(-1, 1)).flatten()

    X_train, y_train = [], []
    for i in range(len(train_scaled) - TIMESTAMP):
        X_train.append(train_scaled[i : i + TIMESTAMP])
        y_train.append(train_scaled[i + TIMESTAMP])

    X_train = np.array(X_train).reshape(-1, TIMESTAMP, 1)
    y_train = np.array(y_train)

    # 모델 아예 재학습
    from keras.models import Sequential # type: ignore
    from keras.layers import Dense, LSTM, Dropout # type: ignore

    model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(TIMESTAMP, 1), activation='tanh'),
                Dropout(0.2),
                LSTM(64, return_sequences=True, activation='tanh'),
                LSTM(32, return_sequences=True, activation='tanh'),
                LSTM(16,  return_sequences=False, activation='tanh'),
                Dense(1)
             ])
    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # 모델 다시 저장
    model.save(MODEL_SAVE_PATH[col])

    return process_(dataset, col=col, threshold=THRESHOLD[col], scaler=scaler)


# 진행 함수 
def process(dataset):
    cols = [col for col in dataset.columns if col in THRESHOLD]

    
    return_dict = {}
    
    # overall 위한 처리
    predicts = []
    return_dict['token_count'] = 0
    for col in cols:
        print(f"====== {col} ======")
        _, graph_LSTM, predict_value, next_time_pred = process_(dataset, col, threshold=THRESHOLD[col])
        key = f"{col.lower()}_graph"
        return_dict[key] = graph_LSTM
        
        predicts.append(predict_value)
        return_dict['token_count'] += next_time_pred
    
    # overall - sum
    start_date = dataset[PREDICT_START:].index[TIMESTAMP] 
    end_date = dataset.index[-1] + pd.Timedelta(days=HORIZON)
    future_index = pd.date_range(start=start_date, end=end_date, freq='D', name='date')

    sum_predict = [sum(vals) for vals in zip(*predicts)]
    sum_predict = pd.Series(sum_predict, index=future_index)

    sum_real = dataset[start_date:][cols].sum(axis=1)
    return_dict['overall_graph'] = plot_predictions(sum_real, sum_predict, col='All token')
    
    return return_dict


# 추가된 함수: 모델 아키텍처 이미지 경로 반환
def get_model_shapes_png():
    """모델 구조 이미지의 경로 반환"""
    return MODEL_SHAPES_PLOT_PATH

# 추가된 함수: 예측 이미지 경로 반환
def get_stock_png():
    """주식 예측 결과 이미지의 경로 반환"""
    return PREDICTION_PLOT_PATH
