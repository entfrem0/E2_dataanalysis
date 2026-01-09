# improved.py (FP-7)
# 指示:
# - FP-6の「2つの main results」を両方改善して報告する
# - 改善は preprocessing / dimensionality reduction (PCA) のみ
# - validationは単発splitではなく、RepeatedKFoldで分布(平均±SD)を出す

from __future__ import annotations

import pandas as pd

from machine import build_pipeline, run_cv_regression, boxplot_scores


def run_fp7_improvements(
    df: pd.DataFrame,
    features: list[str],
    target: str = "dvdt",
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 0,
    show_plots: bool = True,
) -> dict:
    """
    FP-6の2つのmain resultを想定:
      Result 1: KNR 回帰
      Result 2: SVR 回帰

    それぞれについて
      baseline: StandardScaler + model
      improved: StandardScaler + PCA(0.95) + model
    を比較し、CV分布(平均±SD)を返す
    """

    results = {}

    for model_type in ["knr", "svr"]:
        # --- baseline ---
        pipe_base = build_pipeline(
            model_type=model_type,
            scaler="standard",
            use_pca=False,
            random_state=random_state,
        )
        base_scores, base_summary = run_cv_regression(
            df=df,
            features=features,
            target=target,
            pipeline=pipe_base,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        # --- improved (PCA) ---
        pipe_pca = build_pipeline(
            model_type=model_type,
            scaler="standard",
            use_pca=True,
            pca_variance=0.95,
            random_state=random_state,
        )
        pca_scores, pca_summary = run_cv_regression(
            df=df,
            features=features,
            target=target,
            pipeline=pipe_pca,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        results[model_type] = {
            "baseline_scores": base_scores,
            "baseline_summary": base_summary,
            "pca_scores": pca_scores,
            "pca_summary": pca_summary,
        }

        if show_plots:
            boxplot_scores(base_scores, f"{model_type.upper()} baseline CV scores")
            boxplot_scores(pca_scores, f"{model_type.upper()} + PCA(0.95) CV scores")

    return results


def summarize_fp7(results: dict) -> pd.DataFrame:
    """
    KNR/SVRそれぞれの baseline vs PCA の mean/std を1枚にまとめる
    """
    rows = []
    for model_type, d in results.items():
        for label in ["baseline", "pca"]:
            summary = d[f"{label}_summary"]
            row_mean = summary.loc["mean"].to_dict()
            row_std = summary.loc["std"].to_dict()
            rows.append(
                {
                    "model": model_type,
                    "variant": label,
                    "r2_mean": row_mean["r2"],
                    "r2_std": row_std["r2"],
                    "rmse_mean": row_mean["rmse"],
                    "rmse_std": row_std["rmse"],
                    "mae_mean": row_mean["mae"],
                    "mae_std": row_std["mae"],
                }
            )
    return pd.DataFrame(rows)