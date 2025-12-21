# machine.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 回帰モデルを実行する関数
# --------------------------------------------------
def run_regression(
    df,
    features=['u', 'v', 'p'],
    target='dvdt',
    model_type='knr',
    test_size=0.2,
    random_state=42
):
    """
    KNR または SVR による回帰を実行し，
    予測結果と評価指標を返す
    """

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # モデル選択
    if model_type == 'knr':
        model = KNeighborsRegressor(
            n_neighbors=20,
            weights='distance'
        )
        model_name = 'KNR'

    elif model_type == 'svr':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=10, gamma='scale'))
        ])
        model_name = 'SVR'

    else:
        raise ValueError("model_type は 'knr' または 'svr'")

    # 学習
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)

    # 結果まとめ
    results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'flow_type': df.loc[y_test.index, 'flow_type']
    })

    metrics = {
        'model': model_name,
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return results, metrics


# --------------------------------------------------
# 予測結果の可視化
# --------------------------------------------------
def plot_prediction(results, title):
    """
    真値 vs 予測値 の散布図を描画
    """

    plt.figure(figsize=(5, 5))
    sns.scatterplot(
        data=results,
        x='y_true',
        y='y_pred',
        hue='flow_type',
        alpha=0.4
    )

    lims = [
        results[['y_true', 'y_pred']].min().min(),
        results[['y_true', 'y_pred']].max().max()
    ]

    plt.plot(lims, lims, 'k--', lw=2)
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()
