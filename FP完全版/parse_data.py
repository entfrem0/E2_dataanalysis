import pandas as pd
from IPython.display import display


def display_title(title, pref='Table', num=1, center=False):
    """Display a formatted title for tables or figures."""
    text = f"{pref} {num}: {title}"
    if center:
        print(text.center(80))
    else:
        print(text)


def dispersion(x: pd.Series) -> pd.Series:
    """Calculate dispersion-related statistics for a numeric column."""
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


def display_dispersion_table(df: pd.DataFrame, num: int = 1) -> None:
    """Display dispersion statistics for all numeric columns in a DataFrame."""

    # タイトル表示関数
    display_title(
        'Dispersion summary statistics.',
        pref='Table',
        num=num,
        center=False
    )

    # 数値列選択
    numeric_df = df.select_dtypes(include='number')

    # dispersion を列ごとに適用
    df_dispersion = numeric_df.apply(dispersion, axis=0)

    # 行ラベル
    df_dispersion.index = [
        'st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR'
    ]

    display(df_dispersion)