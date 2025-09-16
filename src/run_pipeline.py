from pathlib import Path
from utils_io import load_config, ensure_dir
from preprocess_hypoxia import load_hypoxia_means, filter_by_sessions, long_from_blocks, compute_slopes_1to4
from preprocess_gps import read_gps, clean_gps, summarize_gps_by_player
from merge_and_positions import add_positions
from models_and_plots import violin_paired_first_last, ols_vs_mixed, add_position_and_run_by_position
import pandas as pd
import numpy as np

def main():
    cfg = load_config()
    ensure_dir(cfg["outputs"]["figures_dir"])
    ensure_dir(cfg["outputs"]["tables_dir"])

    # Hypoxia
    mean = load_hypoxia_means(cfg["paths"]["excel_hypoxia"])
    hyp_ols = filter_by_sessions(mean, cfg["analysis"]["min_sessions_for_ols"])
    hyp_mixed = filter_by_sessions(mean, cfg["analysis"]["min_sessions_for_mixed"])

    # Long tables
    long_ols = long_from_blocks(hyp_ols, cfg["analysis"]["use_blocks_for_ols"])
    long_mixed = long_from_blocks(hyp_mixed, cfg["analysis"]["mixed_blocks"])

    # Slopes 1–4
    slopes = compute_slopes_1to4(hyp_ols)

    # GPS
    gps_raw = read_gps(cfg["paths"]["gps_csv"])
    gps = clean_gps(gps_raw)
    gps_sum = summarize_gps_by_player(gps)
    gps_sum.to_csv(Path(cfg["outputs"]["tables_dir"], "gps_summary_by_player.csv"), index=False)

    # Violin paired: block1 vs last (build paired from hyp_ols)
    paired = []
    for _, r in hyp_ols.iterrows():
        if not pd.isna(r.get("SPO2_1")):
            last_j = max([j for j in range(1,10) if f"SPO2_{j}" in hyp_ols.columns and not pd.isna(r.get(f"SPO2_{j}"))])
            paired.append({"Jugador": r["Nombre"], "SPO2_block1": r["SPO2_1"], "SPO2_last": r[f"SPO2_{last_j}"]})
    paired_df = pd.DataFrame(paired)
    violin_paired_first_last(paired_df, Path(cfg["outputs"]["figures_dir"], "violin_paired_spo2.png"))

    # OLS vs Mixed
    ols_vs_mixed(long_mixed, "Observed vs OLS vs Mixed (blocks 1–6)",
                 Path(cfg["outputs"]["figures_dir"], "ols_vs_mixed_1_6.png"))

    # Positions + adjustment
    gp, model = add_position_and_run_by_position(slopes, gps_sum, cfg["paths"]["positions_csv"],
                                                 out_table=Path(cfg["outputs"]["tables_dir"], "merged_slopes_gps_positions.csv"),
                                                 fig_out=Path(cfg["outputs"]["figures_dir"], "slope_by_line.png"))
    if model is not None:
        with open(Path(cfg["outputs"]["tables_dir"], "ols_adjusted_by_position.txt"), "w") as f:
            f.write(model.summary().as_text())

if __name__ == "__main__":
    main()
