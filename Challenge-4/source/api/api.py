from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pydantic import BaseModel

app = FastAPI()

# Carregar o modelo e scaler
model = tf.keras.models.load_model('../models/lstm_model.h5')
scaler = joblib.load('../models/scaler.pkl')

class StockData(BaseModel):
    dates: list
    close_prices: list

@app.post("/predict/")
async def predict(data: StockData):
    df = pd.DataFrame({'Date': pd.to_datetime(data.dates).astype('int64'), 'Close': data.close_prices})
    dado_escalado = scaler.transform(df[['Date', 'Close']])

    janela = 60
    x_lstm = []
    for i in range(janela, len(dado_escalado)):
        x_lstm.append(dado_escalado[i-janela:i, :])
    x_lstm = np.array(x_lstm)

    predictions = model.predict(x_lstm)
    predictions_original = scaler.inverse_transform(np.column_stack((x_lstm[:, :, 0], predictions[:, 0])))[:, 1]
    return {"predictions": predictions_original.tolist()}