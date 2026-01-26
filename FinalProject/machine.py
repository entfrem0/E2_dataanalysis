# machine.py (FP-7)
# - FP-6と同じモデル（KNR, SVR）を扱える
# - FP-7向けに RepeatedKFold CV で分布を出す
# - preprocessing / PCA を Pipeline でON/OFFできる

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_pipeline(
    model_type: str,
    scaler: str = "standard",
    use_pca: bool = False,
    pca_variance: float = 0.95,
    random_state: int = 0,
) -> Pipeline:
    """
    FP-7: preprocessing / PCA のみで改善するためのパイプライン生成
    model_type: "knr" or "svr"
    scaler: "standard" or "robust" or "none"
    use_pca: True/False
    """
    steps = []

    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler == "none":
        pass
    else:
        raise ValueError("scaler must be one of: 'standard', 'robust', 'none'")

    if use_pca:
        # n_components に 0-1 を指定すると “説明分散比で自動次元数”
        steps.append(("pca", PCA(n_components=pca_variance, random_state=random_state)))

    model_type = model_type.lower()
    if model_type == "knr":
        model = KNeighborsRegressor(n_neighbors=15, weights="distance")
    elif model_type == "svr":
        model = SVR(C=10.0, epsilon=0.1, gamma="scale")
    else:
        raise ValueError("model_type must be 'knr' or 'svr'")

    steps.append(("model", model))
    return Pipeline(steps)


def run_cv_regression(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    pipeline: Pipeline,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    RepeatedKFold CVでスコア分布を返す
    returns:
      - scores_df: 各foldの r2/rmse/mae
      - summary_df: mean/std の2行
    """
    X = df[features]
    y = df[target]

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    scoring = {
        "r2": "r2",
        "rmse": make_scorer(_rmse, greater_is_better=False),  # negになる
        "mae": "neg_mean_absolute_error",
    }

    out = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

    scores_df = pd.DataFrame(
        {
            "r2": out["test_r2"],
            "rmse": -out["test_rmse"],
            "mae": -out["test_mae"],
        }
    )

    summary_df = scores_df.agg(["mean", "std"])
    return scores_df, summary_df


def boxplot_scores(scores_df: pd.DataFrame, title: str) -> None:
    """
    CVスコア分布の箱ひげ図（色指定なし）
    """
    ax = scores_df[["r2", "rmse", "mae"]].plot(kind="box", title=title)
    ax.set_ylabel("score")
    plt.show()
    
def run_baseline_models_cv(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_types: list[str] = ["knr", "svr"],
    scaler: str = "standard",
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 0,
    show_plots: bool = True,
    print_summary: bool = True,
) -> dict:
    """
    FP-6の2本（例: KNR/SVR）baselineをCVでまとめて実行して、
    各モデルの scores と summary を返す。
    """
    if print_summary:
        print("=== Baseline (FP-6 models) with CV ===")

    results = {}
    for model_type in model_types:
        pipe = build_pipeline(
            model_type=model_type,
            scaler=scaler,
            use_pca=False,
            random_state=random_state,
        )

        scores, summary = run_cv_regression(
            df=df,
            features=features,
            target=target,
            pipeline=pipe,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        results[model_type] = {"scores": scores, "summary": summary}

        if print_summary:
            print(f"\n--- {model_type.upper()} baseline ({scaler}) ---")
            print(summary)

        if show_plots:
            boxplot_scores(scores, f"{model_type.upper()} baseline CV scores")

    return results