# # model.py
# # -*- coding: utf-8 -*-
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential, load_model
# from keras.layers import Dense, LSTM, Dropout
# from keras.utils import plot_model
# import os
# import math
# from sklearn.metrics import mean_squared_error
# from config import MODEL_SAVE_PATH, MODEL_ARCHI_PLOT_PATH, MODEL_SHAPES_PLOT_PATH, \
#                    TIMESTAMP, PREDICT_START, DATA_PATH

# # 데이터 로딩
# dataset = pd.read_csv(DATA_PATH, index_col='date', parse_dates=['date'], encoding='utf-8')

# for col in list(dataset.columns):
#     print(f'============== {col} ==============')

#     training_set = pd.DataFrame(dataset[:'2023'][col]).values
#     test_set = pd.DataFrame(dataset[PREDICT_START:][col]).values

#     # 학습 데이터 전체 스케일링
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     train_scaled = scaler.fit_transform(training_set.reshape(-1, 1)).flatten()

#     # 학습용 윈도우 생성
#     X_train, y_train = [], []
#     for i in range(len(train_scaled) - TIMESTAMP):
#         X_train.append(train_scaled[i : i + TIMESTAMP])
#         y_train.append(train_scaled[i + TIMESTAMP])
#     X_train = np.array(X_train).reshape(-1, TIMESTAMP, 1)
#     y_train = np.array(y_train)

#     # 모델 정의
#     model = Sequential([
#         LSTM(128, return_sequences=True, input_shape=(TIMESTAMP, 1), activation='tanh'),
#         Dropout(0.2),
#         LSTM(64, return_sequences=True, activation='tanh'),
#         LSTM(32, return_sequences=True, activation='tanh'),
#         LSTM(16,  return_sequences=False, activation='tanh'),
#         Dense(1)
#     ])
#     model.compile(optimizer='rmsprop', loss='mean_squared_error')

#     # 모델 학습
#     model.fit(X_train, y_train, epochs=50, batch_size=32)


#     # 테스트 데이터 동일 스케일러로 변환
#     test_scaled = scaler.transform(test_set.reshape(-1, 1)).flatten()

#     # 테스트용 윈도우 생성
#     X_test= []
#     for i in range(len(test_scaled) - TIMESTAMP):
#         X_test.append(test_scaled[i : i + TIMESTAMP])
#     X_test = np.array(X_test).reshape(-1, TIMESTAMP, 1)

#     # 예측 & 역정규화
#     y_pred_scaled = model.predict(X_test)
#     y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

#     # RMSE 확인
#     #print(math.sqrt(mean_squared_error(test_set[TIMESTAMP:], y_pred)))

#     # 모델 저장
#     model.save(MODEL_SAVE_PATH[col])


# # 모델 구조 이미지 생성 (모두 동일한 아키이므로 하나만 저장)
# plot_model(model, to_file=MODEL_ARCHI_PLOT_PATH)
# plot_model(model, to_file=MODEL_SHAPES_PLOT_PATH, show_shapes=True)
# print(f"Model structure saved to '{MODEL_ARCHI_PLOT_PATH}' and '{MODEL_SHAPES_PLOT_PATH}'")
