from scaler import scale_data

import numpy as np
import pandas as pd
import joblib

from tensorflow.python.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, epochs=50, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def evaluate_model(model, x_test, y_test, sc):
    y_pred_teste = model.predict(x_test)
    y_test_original = sc.inverse_transform(np.column_stack((x_test[:, :, 0], y_test)))[:, 1]
    y_pred_teste_original = sc.inverse_transform(np.column_stack((x_test[:, :, 0], y_pred_teste[:, 0])))[:, 1]
    mape = np.mean(np.abs((y_test_original - y_pred_teste_original) / y_test_original)) * 100
    mae = mean_absolute_error(y_test_original, y_pred_teste_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_teste_original))
    return mape, mae, rmse, y_pred_teste_original

if __name__ == "__main__":
    df = pd.read_csv('../stock_data.csv')
    df['Date'] = pd.to_datetime(df['Date']).astype('int64') / 10**9
    df_numeric = df[['Date', 'Close']].dropna().astype(float)
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
    df_numeric = df_numeric.dropna()
    dado_escalado, sc = scale_data(df_numeric)
    x = dado_escalado[:, 0]
    y = dado_escalado[:, 1]

    # Criar dados sequenciais para LSTM
    janela = 60
    x_lstm = []
    y_lstm = []
    for i in range(janela, len(x)):
        x_lstm.append(dado_escalado[i-janela:i, :])
        y_lstm.append(dado_escalado[i, 1])

    x_lstm, y_lstm = np.array(x_lstm), np.array(y_lstm)

    tamanho_teste = int(0.1 * len(x_lstm))
    tamanho_treino = len(x_lstm) - tamanho_teste
    x_treino, y_treino = x_lstm[:tamanho_treino], y_lstm[:tamanho_treino]
    x_teste, y_teste = x_lstm[tamanho_treino:], y_lstm[tamanho_treino:]

    model = create_lstm_model((x_treino.shape[1], x_treino.shape[2]))
    model = train_model(model, x_treino, y_treino)
    mape, mae, rmse, y_pred_teste_original = evaluate_model(model, x_teste, y_teste, sc)

    print(f"MAPE: {mape:.2f}%")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Salvar o modelo e scaler
    model.save('../models/lstm_model.h5')
    joblib.dump(sc, '../models/scaler.pkl')

    df_resultados = pd.DataFrame({'y_teste_original': y_teste, 'y_pred_teste': y_pred_teste_original})
    df_resultados.to_excel('../results/previsoes.xlsx', index=False)