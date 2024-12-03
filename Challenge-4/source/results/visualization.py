import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_data(df, title):
    plt.figure(figsize=(20, 6))
    plt.rcParams['font.size'] = 10
    sns.lineplot(x='Date', y='Close', data=df)
    plt.xlabel('Período em Dias')
    plt.ylabel('Cotação de Fechamento')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('../data/stock_data.csv')
    plot_data(df, title='Histórico de Preços')