# improved.py (FP-7)
# 指示:
# - FP-6の「2つの main results」を両方改善して報告する
# - 改善は preprocessing / dimensionality reduction (PCA) のみ
# - validationは単発splitではなく、RepeatedKFoldで分布(平均±SD)を出す

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from machine import build_pipeline, run_cv_regression, boxplot_scores, run_baseline_models_cv


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

from machine import build_pipeline, run_cv_regression, boxplot_scores

def run_improved_models_cv(
    df,
    features,
    target,
    model_types=["knr", "svr"],
    scaler="standard",
    pca_variance=0.95,
    n_splits=5,
    n_repeats=10,
    random_state=0,
    show_plots=True,
    print_summary=True,
):
    """
    FP-7 Improved: preprocessing + PCA のみを追加したパイプラインで
    KNR/SVR をまとめてCV評価する。
    各モデルの scores と summary を返す。
    """
    if print_summary:
        print("=== Improved (preprocessing + PCA) with CV ===")

    results = {}
    for model_type in model_types:
        pipe = build_pipeline(
            model_type=model_type,
            scaler=scaler,
            use_pca=True,
            pca_variance=pca_variance,
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
            print(f"\n--- {model_type.upper()} improved ({scaler} + PCA({pca_variance})) ---")
            print(summary)

        if show_plots:
            boxplot_scores(scores, f"{model_type.upper()} improved CV scores (PCA {pca_variance})")

    return results





def run_ml_section_compact(
    df,
    features,
    target="dvdt",
    n_splits=5,
    n_repeats=10,
    random_state=0,
    show_plot=True,
    print_summary=True,
):
    """
    Final Project / FP-7 の Machine Learning セクションを
    notebook側を最小化するための“まとめ関数”。

    - baseline（StandardScalerのみ）をCVで評価
    - improved（StandardScaler + PCA）をCVで評価
    - R²の箱ひげ図を1枚にまとめて表示（任意）
    - mean±std の比較表（r2/rmse/mae）をDataFrameで返す

    returns:
      baseline_results, improved_results, summary_table
    """

    # 1) baseline (図は出さない)
    baseline_results = run_baseline_models_cv(
        df=df,
        features=features,
        target=target,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        show_plots=False,
        print_summary=print_summary,
    )

    # 2) improved (図は出さない)
    improved_results = run_improved_models_cv(
        df=df,
        features=features,
        target=target,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        show_plots=False,
        print_summary=print_summary,
    )

    # 3) 1枚の箱ひげ図（R²）
    if show_plot:
        groups = [
            ("KNR baseline", baseline_results["knr"]["scores"]["r2"].values),
            ("KNR + PCA",    improved_results["knr"]["scores"]["r2"].values),
            ("SVR baseline", baseline_results["svr"]["scores"]["r2"].values),
            ("SVR + PCA",    improved_results["svr"]["scores"]["r2"].values),
        ]

        plt.figure(figsize=(10, 5))
        plt.boxplot([g[1] for g in groups], labels=[g[0] for g in groups], showmeans=True)
        plt.ylabel("R² (unitless)")
        plt.title("Cross-validation R² distributions: Baseline vs PCA (one figure)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()

    # 4) mean±std の比較表
    def _summary_row(model_name, variant, summary_df):
        return {
            "model": model_name,
            "variant": variant,
            "r2_mean": float(summary_df.loc["mean", "r2"]),
            "r2_std": float(summary_df.loc["std", "r2"]),
            "rmse_mean": float(summary_df.loc["mean", "rmse"]),
            "rmse_std": float(summary_df.loc["std", "rmse"]),
            "mae_mean": float(summary_df.loc["mean", "mae"]),
            "mae_std": float(summary_df.loc["std", "mae"]),
        }

    summary_table = pd.DataFrame([
        _summary_row("KNR", "baseline", baseline_results["knr"]["summary"]),
        _summary_row("KNR", "PCA",      improved_results["knr"]["summary"]),
        _summary_row("SVR", "baseline", baseline_results["svr"]["summary"]),
        _summary_row("SVR", "PCA",      improved_results["svr"]["summary"]),
    ])

    return baseline_results, improved_results, summary_table