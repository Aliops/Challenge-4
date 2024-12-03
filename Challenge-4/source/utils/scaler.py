import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_data(df):
    sc = StandardScaler()
    dado_escalado = sc.fit_transform(df[['Date', 'Close']])
    return dado_escalado, sc
