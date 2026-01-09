# improved.py

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from machine import run_regression


def evaluate_by_flow_type(results):
    """
    flow_type ごとに R2 と RMSE を計算する
    """
    metrics = []

    for ft in results['flow_type'].unique():
        subset = results[results['flow_type'] == ft]

        r2 = r2_score(subset['y_true'], subset['y_pred'])
        rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))

        metrics.append({
            'flow_type': ft,
            'R2': r2,
            'RMSE': rmse
        })

    return pd.DataFrame(metrics)


def run_and_evaluate(
    df,
    features,
    target='dvdt',
    model_type='knr'
):
    """
    回帰を実行し，全体評価と flow_type 別評価を返す
    """
    results, overall_metrics = run_regression(
        df,
        features=features,
        target=target,
        model_type=model_type
    )

    flow_metrics = evaluate_by_flow_type(results)

    return results, overall_metrics, flow_metrics