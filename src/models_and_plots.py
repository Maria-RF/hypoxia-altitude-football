import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def violin_paired_first_last(paired_df, outpath=None):
    import matplotlib.pyplot as plt
    from scipy import stats
    x1 = paired_df["SPO2_block1"].to_numpy(float)
    x2 = paired_df["SPO2_last"].to_numpy(float)
    t = stats.ttest_rel(x1, x2)
    fig, ax = plt.subplots(figsize=(7,5))
    parts = ax.violinplot([x1, x2], positions=[1,2], showmeans=True, showmedians=True, showextrema=True)
    rng = np.random.default_rng(42)
    j1 = rng.normal(1.0, 0.03, len(x1)); j2 = rng.normal(2.0, 0.03, len(x2))
    ax.scatter(j1, x1, alpha=0.7); ax.scatter(j2, x2, alpha=0.7)
    for a,b,c,d in zip(j1,x1,j2,x2): ax.plot([a,c],[b,d], alpha=0.3)
    ax.set_xticks([1,2]); ax.set_xticklabels(["Block 1","Last block"])
    ax.set_ylabel("SpO₂ (%)")
    ax.set_title(f"SpO₂ paired: t({len(x1)-1})={t.statistic:.2f}, p={t.pvalue:.3f}")
    plt.tight_layout()
    if outpath: plt.savefig(outpath, dpi=200)
    plt.show()

def ols_vs_mixed(long_df, label, outpath=None):
    # OLS
    ols = smf.ols("SpO2 ~ block", data=long_df).fit()
    # Mixed (random intercept + slope)
    mixed = smf.mixedlm("SpO2 ~ block", long_df, groups=long_df["Jugador"], re_formula="~block").fit(method="nm", reml=False, maxiter=1000, full_output=True, disp=False)
    # Observed means
    obs = long_df.groupby("block")["SpO2"].agg(["mean","sem"]).reset_index()
    obs["ci"] = 1.96*obs["sem"]
    x = np.arange(int(long_df.block.min()), int(long_df.block.max())+1)
    y_ols = ols.params["Intercept"] + ols.params["block"]*x
    y_mix = mixed.params["Intercept"] + mixed.params["block"]*x
    plt.figure(figsize=(8,6))
    plt.errorbar(obs["block"], obs["mean"], yerr=obs["ci"], fmt="o", color="black", capsize=5, label="Observed (±95%CI)")
    plt.plot(x, y_ols, "--", label=f"OLS (slope={ols.params['block']:.2f}, p={ols.pvalues['block']:.3f})")
    plt.plot(x, y_mix, "-", label=f"Mixed (slope={mixed.params['block']:.2f}, p={mixed.pvalues['block']:.3f})")
    plt.xlabel("Block"); plt.ylabel("SpO₂ (%)"); plt.title(label); plt.legend(); plt.tight_layout()
    if outpath: plt.savefig(outpath, dpi=200)
    plt.show()
    return ols, mixed

def add_position_and_run_by_position(slopes_df, gps_df, positions_csv, out_table=None, fig_out=None):
    pos = pd.read_csv(positions_csv)
    pos = pos.rename(columns={pos.columns[0]: "Jugador"})  # Name/Nombre → Jugador
    merged = slopes_df.merge(pos[["Jugador","Position","Line"]], on="Jugador", how="left")
    if "Name" in gps_df.columns:
        gps_df = gps_df.rename(columns={"Name":"Jugador"})
    gp = merged.merge(gps_df, on="Jugador", how="left")
    # Example: OLS adjusting for position line (DEF/MID/FWD) predicting m_per_min by slope
    if "m_per_min (avg)" in gp.columns:
        model = smf.ols("Q(`m_per_min (avg)`) ~ slope_1to4 + C(Line)", data=gp).fit()
    else:
        model = None
    # Plot slopes by Line
    plt.figure(figsize=(6,4))
    sns.boxplot(data=gp, x="Line", y="slope_1to4")
    sns.stripplot(data=gp, x="Line", y="slope_1to4", alpha=0.6)
    plt.title("SpO₂ slope (1–4) by line")
    plt.tight_layout()
    if fig_out: plt.savefig(fig_out, dpi=200)
    plt.show()
    if out_table: gp.to_csv(out_table, index=False)
    return gp, model
