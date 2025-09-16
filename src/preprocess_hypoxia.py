import pandas as pd
import numpy as np

def load_hypoxia_means(excel_path):
    mean = pd.read_excel(excel_path, sheet_name="Mean Datos por Sesion")
    resumen = pd.read_excel(excel_path, sheet_name="Resumen")
    mean = mean.merge(resumen[["Nombre","Total de sesiones (n)"]], on="Nombre", how="left")
    return mean

def filter_by_sessions(df, min_sessions):
    return df[df["Total de sesiones (n)"] >= min_sessions].copy()

def long_from_blocks(df, blocks, name_col="Nombre"):
    rows = []
    for _, r in df.iterrows():
        for j, b in enumerate(blocks, start=1):
            col = f"SPO2_{j}"
            if col in df.columns and pd.notnull(r[col]):
                rows.append({"Jugador": r[name_col], "block": j, "SpO2": float(r[col])})
    return pd.DataFrame(rows)

def compute_slopes_1to4(df_means, player_col="Nombre"):
    # simple regression slope across SPO2_1..SPO2_4 for each player
    out = []
    for _, r in df_means.iterrows():
        y = r[[f"SPO2_{i}" for i in range(1,5)]].astype(float).values
        x = np.arange(1,5)
        mask = ~np.isnan(y)
        if mask.sum() > 1:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            out.append({"Jugador": r[player_col], "slope_1to4": slope})
    return pd.DataFrame(out)
