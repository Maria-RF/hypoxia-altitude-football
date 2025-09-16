# scripts/run_fast.py
from pathlib import Path
import re
import unicodedata

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend no interactivo (más rápido/estable)
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

# ----------------------------
# RUTAS (ajusta si es necesario)
# ----------------------------
EXCEL = "data/DataBase_Futbol_2025v.xlsx"
CSV   = "data/export_15-09-25.csv"

OUT_FIG = Path("outputs/figures")
OUT_TAB = Path("outputs/tables")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Utils
# ----------------------------
def canonical_key(s: str) -> str:
    """Normaliza a minúsculas, sin tildes/ñ y sin signos (p. ej., 'Víctor Dávila' → 'victor davila')."""
    if s is None:
        return ""
    s = str(s).casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Diccionario para corregir typos/variantes
aliases = {
    canonical_key("Igor Linchnovsky"): canonical_key("Igor Lichnovsky"),
    canonical_key("Paulo Diaz"): canonical_key("Paulo Díaz"),
    canonical_key("Alexis Sanchez"): canonical_key("Alexis Sánchez"),
    canonical_key("Marcelino Nunez"): canonical_key("Marcelino Núñez"),
    canonical_key("Victor Davila"): canonical_key("Víctor Dávila"),
    canonical_key("Rodrigo Echeverria"): canonical_key("Rodrigo Echeverría"),
    canonical_key("Fabian Hormazabal"): canonical_key("Fabián Hormazábal"),
    canonical_key("Benjamin Kuscevic"): canonical_key("Benjamín Kuscevic"),
    canonical_key("Rodrigo Urena"): canonical_key("Rodrigo Ureña"),
    canonical_key("Dario Osorio"): canonical_key("Darío Osorio"),
}
def apply_alias(k: str) -> str:
    return aliases.get(k, k)

def ensure_cols(df: pd.DataFrame, cols_defaults: dict) -> pd.DataFrame:
    """Garantiza que existan columnas con un valor por defecto."""
    for c, default in cols_defaults.items():
        if c not in df.columns:
            df[c] = default
    return df

# ---------------------------------------------------
# 1) HIPÓXIA: VIOLÍN (SpO2 bloque 1 vs último bloque)
# ---------------------------------------------------
mean = pd.read_excel(EXCEL, sheet_name="Mean Datos por Sesion")
res  = pd.read_excel(EXCEL, sheet_name="Resumen")

# Añade número de sesiones
mean = mean.merge(res[["Nombre","Total de sesiones (n)"]], on="Nombre", how="left")
mean4 = mean[mean["Total de sesiones (n)"]>=4].copy()

paired = []
for _, r in mean4.iterrows():
    if pd.notna(r.get("SPO2_1")):
        last_j = max([j for j in range(1, 50)
                      if f"SPO2_{j}" in mean4.columns and pd.notna(r.get(f"SPO2_{j}"))])
        paired.append({
            "Jugador": r["Nombre"],
            "SPO2_block1": float(r["SPO2_1"]),
            "SPO2_last": float(r[f"SPO2_{last_j}"])
        })
paired_df = pd.DataFrame(paired)

if len(paired_df) >= 2:
    x1 = paired_df["SPO2_block1"].to_numpy(float)
    x2 = paired_df["SPO2_last"].to_numpy(float)
    t = stats.ttest_rel(x1, x2)

    plt.figure(figsize=(7,5))
    plt.violinplot([x1, x2], positions=[1,2], showmeans=True, showmedians=True, showextrema=True)
    rng = np.random.default_rng(42)
    j1 = rng.normal(1.0, 0.03, len(x1))
    j2 = rng.normal(2.0, 0.03, len(x2))
    plt.scatter(j1, x1, alpha=0.7)
    plt.scatter(j2, x2, alpha=0.7)
    for a, b, c, d in zip(j1, x1, j2, x2):
        plt.plot([a, c], [b, d], alpha=0.3)
    plt.xticks([1,2], ["Block 1","Last block"])
    plt.ylabel("SpO₂ (%)")
    plt.title(f"SpO₂ paired: t({len(x1)-1})={t.statistic:.2f}, p={t.pvalue:.3f}")
    plt.tight_layout()
    plt.savefig(OUT_FIG/"violin_paired_spo2.png", dpi=200)
    plt.close()

# ---------------------------------------------------
# 2) GPS: RESUMEN RÁPIDO + GRÁFICO m/min vs MAI/min
# ---------------------------------------------------
# Lectura robusta del CSV
df = pd.read_csv(CSV, engine="python", encoding="utf-8-sig")

# Renombres mínimos a nombres estándar
rename = {
    "Name":"name",
    "Minutos":"minutes",
    "Total Distance (m)":"distance_m",
    "Mts/min":"m_per_min",
    "MAInt >20km/h":"mai_m",
    "AInt >15km/h":"hsr_m",
    "MAI/min":"mai_per_min",
    "SPRINT>25km/h":"sprint_m",
    # Si vinieran ya en el CSV
    "Position":"Position",
    "Line":"Line",
}
df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

# Derivar per-90 si hay minutos
if "minutes" in df.columns:
    m = df["minutes"].replace(0, np.nan)
    for base, out in [("distance_m","dist_per90"),
                      ("hsr_m","hsr_per90"),
                      ("sprint_m","sprint_per90"),
                      ("mai_m","mai_per90")]:
        if base in df.columns:
            df[out] = df[base] * (90.0 / m)

# Guardar resumen por jugador (solo columnas disponibles)
keep = [c for c in [
    "name","minutes","distance_m","m_per_min","hsr_m","sprint_m","mai_m",
    "dist_per90","hsr_per90","sprint_per90","mai_per90","mai_per_min",
    "Position","Line"
] if c in df.columns]
summary = df[keep].copy()

# Claves canónicas + alias en summary
summary["name_key"] = summary["name"].apply(canonical_key).apply(apply_alias)

# Fallback 1: completar Position/Line desde data/positions.csv si existe
pos_path = Path("data/positions.csv")
if pos_path.exists():
    pos_df = pd.read_csv(pos_path)
    if "Name" in pos_df.columns and "name" not in pos_df.columns:
        pos_df = pos_df.rename(columns={"Name":"name"})
    pos_df = ensure_cols(pos_df, {"Position":"", "Line":""})
    pos_df["name_key"] = pos_df["name"].apply(canonical_key).apply(apply_alias)
    summary = summary.merge(
        pos_df[["name_key","Position","Line"]],
        on="name_key", how="left", suffixes=("","_pos")
    )
    # <<< FIX: usar combine_first en vez de fillna con Series >>>
    for col in ("Position","Line"):
        if f"{col}_pos" in summary.columns:
            summary[col] = summary.get(col, pd.Series(index=summary.index, dtype=object)).combine_first(summary[f"{col}_pos"])
            summary.drop(columns=[f"{col}_pos"], inplace=True)

# Fallback 2: si aún faltara, intenta desde el Excel 'Resumen' si hay columnas afines
pos_cols_candidates = {
    "Position": ["Position","Posicion","Posición"],
    "Line":     ["Line","Linea","Línea"]
}
rename_map = {}
for std_col, cands in pos_cols_candidates.items():
    for c in cands:
        if c in res.columns:
            rename_map[c] = std_col
            break
if rename_map:
    res_pos = res.rename(columns=rename_map).copy()
    name_col = "Nombre" if "Nombre" in res_pos.columns else ("Name" if "Name" in res_pos.columns else None)
    if name_col:
        cols = [name_col] + [c for c in ["Position","Line"] if c in res_pos.columns]
        pos2 = res_pos[cols].copy().rename(columns={name_col: "name"})
        pos2["name_key"] = pos2["name"].apply(canonical_key).apply(apply_alias)
        summary = summary.merge(
            pos2[["name_key"] + [c for c in ["Position","Line"] if c in pos2.columns]],
            on="name_key", how="left", suffixes=("","_res")
        )
        # <<< FIX: usar combine_first aquí también >>>
        for col in ("Position","Line"):
            if f"{col}_res" in summary.columns:
                summary[col] = summary.get(col, pd.Series(index=summary.index, dtype=object)).combine_first(summary[f"{col}_res"])
                summary.drop(columns=[f"{col}_res"], inplace=True)

# Garantiza Position/Line
summary = ensure_cols(summary, {"Position":"Unknown", "Line":"Unknown"})
for col in ("Position","Line"):
    summary[col] = (summary[col]
                    .astype("string")
                    .fillna("Unknown")
                    .str.strip()
                    .replace({"": "Unknown"}))

# Export resumen
summary.to_csv(OUT_TAB/"gps_summary_by_player.csv", index=False)

# --- Filtrado y alias para hipoxia (limitado a nombres del partido) ---
gps_names_key = set(summary["name_key"])
mean_filtered = mean[mean["Nombre"].apply(canonical_key).apply(apply_alias).isin(gps_names_key)].copy()

# Excluir filas no-jugador (ej. "Promedio/Equipo")
to_exclude = {"promedio", "equipo", "avg", "average"}
mean_filtered = mean_filtered[~mean_filtered["Nombre"].apply(lambda x: canonical_key(x) in to_exclude)]

# Clave canónica + alias en mean_filtered
mean_filtered["name_key"] = mean_filtered["Nombre"].apply(canonical_key).apply(apply_alias)

# ---------------------------------------------------
# Gráficos m/min vs MAI/min
# ---------------------------------------------------
if {"name","m_per_min","mai_per_min","Position","Line"}.issubset(summary.columns):
    plot_df = summary[["name","m_per_min","mai_per_min","Position","Line"]].copy()
    for col in ("Position","Line"):
        plot_df[col] = plot_df[col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})

    plot_df_basic = plot_df.dropna(subset=["m_per_min","mai_per_min"])

    # (1) Scatter básico
    if len(plot_df_basic) >= 2:
        plt.figure(figsize=(7,5))
        plt.scatter(plot_df_basic["m_per_min"], plot_df_basic["mai_per_min"], alpha=0.85)
        x = plot_df_basic["m_per_min"].to_numpy(float)
        y = plot_df_basic["mai_per_min"].to_numpy(float)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        xx = np.linspace(x.min(), x.max(), 100)
        yy = slope*xx + intercept
        plt.plot(xx, yy, linestyle="--", label=f"OLS: y={slope:.2f}x+{intercept:.2f}")
        plt.xlabel("m/min"); plt.ylabel("MAI/min")
        plt.title("Relación m/min vs MAI/min (partido)")
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT_FIG/"m_permin_vs_mai_permin_basico.png", dpi=200); plt.close()

    # (2) Scatter con nombres
    if len(plot_df_basic) >= 1:
        plt.figure(figsize=(8,6))
        plt.scatter(plot_df_basic["m_per_min"], plot_df_basic["mai_per_min"], alpha=0.7)
        for _, r in plot_df_basic.iterrows():
            plt.annotate(str(r["name"]), (r["m_per_min"], r["mai_per_min"]), fontsize=8, alpha=0.85)
        if len(plot_df_basic) >= 2:
            x = plot_df_basic["m_per_min"].to_numpy(float)
            y = plot_df_basic["mai_per_min"].to_numpy(float)
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 100)
            yy = slope*xx + intercept
            plt.plot(xx, yy, linestyle="--", label=f"OLS: y={slope:.2f}x+{intercept:.2f}")
        plt.xlabel("m/min"); plt.ylabel("MAI/min")
        plt.title("Relación m/min vs MAI/min (con nombres)")
        if len(plot_df_basic) >= 2: plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_FIG/"m_permin_vs_mai_permin_labels.png", dpi=200); plt.close()

    # (3) Scatter por línea
    if len(plot_df_basic) >= 1:
        lines = plot_df_basic["Line"].unique().tolist()
        plt.figure(figsize=(8,6))
        for ln in lines:
            sub = plot_df_basic[plot_df_basic["Line"] == ln]
            plt.scatter(sub["m_per_min"], sub["mai_per_min"], alpha=0.85, label=str(ln))
        if len(plot_df_basic) >= 2:
            x = plot_df_basic["m_per_min"].to_numpy(float)
            y = plot_df_basic["mai_per_min"].to_numpy(float)
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 100)
            yy = slope*xx + intercept
            plt.plot(xx, yy, linestyle="--", label=f"OLS global: y={slope:.2f}x+{intercept:.2f}")
        plt.xlabel("m/min"); plt.ylabel("MAI/min")
        plt.title("Relación m/min vs MAI/min por línea (DEF/MID/FWD)")
        plt.legend(title="Línea"); plt.tight_layout()
        plt.savefig(OUT_FIG/"m_permin_vs_mai_permin_por_linea.png", dpi=200); plt.close()

        # aviso de Unknown
        unknowns = plot_df.loc[plot_df["Line"] == "Unknown", "name"].unique().tolist()
        if unknowns:
            print("⚠️ Jugadores con línea 'Unknown' en el gráfico:", ", ".join(unknowns))
        else:
            print("✅ Ningún jugador quedó con línea 'Unknown' en el gráfico.")
else:
    print("Aviso: faltan columnas en summary (name, m_per_min, mai_per_min).")

# ---------------------------------------------------
# 3) Scatter m/min vs SpO₂ (última sesión) coloreado por línea
# ---------------------------------------------------
# Última SpO₂ disponible por jugador
spo2_cols_all = [c for c in mean_filtered.columns if str(c).startswith("SPO2_")]
last_records = []
for _, row in mean_filtered.iterrows():
    valid = [(c, row[c]) for c in spo2_cols_all if pd.notna(row.get(c))]
    if valid:
        last_col, last_val = valid[-1]
        last_records.append({
            "name_key": row["name_key"],
            "name_excel": row["Nombre"],
            "SPO2_last": float(last_val)
        })
spo2_last_df = pd.DataFrame(last_records)

# Merge SpO2_last ←→ GPS (m/min)
merged = pd.merge(
    spo2_last_df,
    summary[["name_key","name","m_per_min","Line"]],
    on="name_key", how="left"
)

# Normalizar Line
merged["Line"] = merged["Line"].fillna("").astype(str).str.strip().replace({"": "Unknown"})
ok = merged.dropna(subset=["m_per_min","SPO2_last"]).copy()

# Mensajes de diagnóstico
faltan_gps = merged[merged["m_per_min"].isna()]["name_excel"].unique().tolist()
if faltan_gps:
    print("⚠️ Aún sin m/min (no están en CSV del partido o nombre distinto):", ", ".join(faltan_gps))

# Scatter por línea
if len(ok) >= 2:
    plt.figure(figsize=(8,6))
    for ln in ok["Line"].unique():
        sub = ok[ok["Line"] == ln]
        plt.scatter(sub["m_per_min"], sub["SPO2_last"], alpha=0.85, label=str(ln))
    x = ok["m_per_min"].to_numpy(float); y = ok["SPO2_last"].to_numpy(float)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    xx = np.linspace(x.min(), x.max(), 100); yy = slope*xx + intercept
    r, pval = pearsonr(x, y)
    plt.plot(xx, yy, linestyle="--", label=f"OLS global: y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel("m/min (partido)"); plt.ylabel("SpO₂ última sesión (%)")
    plt.title(f"m/min vs SpO₂ última sesión por línea (n={len(ok)})\nPearson r={r:.2f}, p={pval:.3f}")
    plt.legend(title="Línea"); plt.tight_layout()
    plt.savefig(OUT_FIG/"corr_mmin_spo2_last_by_line.png", dpi=200); plt.close()
    print("✅ Figura actualizada:", OUT_FIG/"corr_mmin_spo2_last_by_line.png")
else:
    print("⚠️ Datos insuficientes para el scatter por línea (se necesitan ≥2 pares).")
