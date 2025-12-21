import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, norm, levene

sns.set(style="whitegrid", font_scale=1.2)


# --------------------------------------------------------
# 1. Pressure vs y-acceleration + ANCOVA
# --------------------------------------------------------
def analysis_pressure_acceleration(df):
    """
    Main Result ①
    Plot p–dvdt for laminar/turbulent and run ANCOVA.
    Returns: model summary (string)
    """

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x='p', y='dvdt', hue='flow_type', alpha=0.4)

    # 回帰線（各flowごと）
    for flow in ['laminar', 'turbulent']:
        subset = df[df['flow_type'] == flow]
        sns.regplot(data=subset, x='p', y='dvdt', scatter=False, label=f"{flow} fit")

    plt.title("Pressure vs y-acceleration (p–dvdt)")
    plt.legend()
    plt.show()

    # ANCOVA
    model = smf.ols('dvdt ~ p * flow_type', data=df).fit()
    return model.summary().as_text()



# --------------------------------------------------------
# 2. Velocity vs Pressure + Fisher r-test
# --------------------------------------------------------
def analysis_velocity_pressure(df):
    """
    Main Result ②
    Plot u–p for laminar/turbulent and test correlation difference.
    Returns: dictionary with r_l, r_t, z, p
    """

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x='u', y='p', hue='flow_type', alpha=0.4)

    for flow in ['laminar', 'turbulent']:
        subset = df[df['flow_type'] == flow]
        sns.regplot(data=subset, x='u', y='p', scatter=False, label=f"{flow} fit")

    plt.title("Velocity vs Pressure (u–p)")
    plt.legend()
    plt.show()

    # 相関係数
    df_l = df[df.flow_type == 'laminar']
    df_t = df[df.flow_type == 'turbulent']

    r_l, _ = pearsonr(df_l['u'], df_l['p'])
    r_t, _ = pearsonr(df_t['u'], df_t['p'])

    # Fisher r→z
    def fisher_z(r):
        return 0.5 * np.log((1 + r) / (1 - r))

    z_l = fisher_z(r_l)
    z_t = fisher_z(r_t)

    n_l = len(df_l)
    n_t = len(df_t)

    z = (z_l - z_t) / np.sqrt(1/(n_l - 3) + 1/(n_t - 3))
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return {
        "r_laminar": r_l,
        "r_turbulent": r_t,
        "z_score": z,
        "p_value": p_value
    }



# --------------------------------------------------------
# 3. Velocity vs Acceleration variance comparison（Levene）
# --------------------------------------------------------
def analysis_acceleration_variance(df):
    """
    classical.ipynb ①
    Compare variance of dudt (laminar vs turbulent)
    Returns: dictionary with var_l, var_t, stat, p
    """

    d_l = df[df.flow_type == 'laminar']['dudt']
    d_t = df[df.flow_type == 'turbulent']['dudt']

    var_l = np.var(d_l, ddof=1)
    var_t = np.var(d_t, ddof=1)

    stat, p = levene(d_l, d_t, center='median')

    return {
        "variance_laminar": var_l,
        "variance_turbulent": var_t,
        "levene_statistic": stat,
        "p_value": p
    }