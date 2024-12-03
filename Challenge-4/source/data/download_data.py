import yfinance as yf
import pandas as pd
import os

def download_data(acao, start, end):
    df = yf.download(acao, start=start, end=end)
    df = df[['Close']].dropna().reset_index()
    df['Date'] = pd.to_datetime(df['Date']).astype('int64')
    return df

if __name__ == "__main__":
    acao = "KNCR11.SA"
    df = download_data(acao, start="2021-01-01", end="2025-12-01")
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'stock_data.csv'), index=False)