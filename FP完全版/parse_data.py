import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import linregress
from IPython.display import display


def dispersion(x: pd.Series) -> pd.Series:
    """数値列のばらつき統計量を返す"""
    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    return pd.Series({
        'st.dev.': x.std(),
        'min': x.min(),
        'max': x.max(),
        'range': x.max() - x.min(),
        '25th': q25,
        '75th': q75,
        'IQR': q75 - q25
    })

def display_dispersion_table(df: pd.DataFrame) -> None:
    """DataFrame の数値列に対して dispersion() を適用して表を表示"""
    numeric_df = df.select_dtypes(include='number')
    df_dispersion = numeric_df.apply(dispersion, axis=0)
    df_dispersion.index = ['st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR']
    display(df_dispersion)