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
    if s is None:
        return ""
    s = str(s).casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Diccionario para corregir typos
aliases = {
    canonical_key("Igor Linchnovsky"): canonical_key("Igor Lichnovsky"),
    canonical_key("Paulo Diaz"): canonical_key("Paulo Díaz"),
    canonical_key("Alexis Sanchez"): canonical_key("Alexis Sánchez"),
    # agrega más si detectas otros
}
def apply_alias(k: str) -> str:
    return aliases.get(k, k)

# ---------------------------------------------------
# 1) HIPÓXIA: VIOLÍN (SpO2 bloque 1 vs último bloque)
# ---------------------------------------------------
mean = pd.read_excel(EXCEL, sheet_name="Mean Datos por Sesion")
res  = pd.read_excel(EXCEL, sheet_name="Resumen")
mean = mean.merge(res[["Nombre","Total de sesiones (n)"]], on="Nombre", how="left")
mean4 = mean[mean["Total de sesiones (n)"]>=4].copy()

paired = []
for _, r in mean4.iterrows():
    if pd.notna(r.get("SPO2_1")):
        last_j = max([j for j in range(1,10)
                      if f"SPO2_{j}" in mean4.columns and pd.notna(r.get(f"SPO2_{j}"))])
        paired.append({
            "Jugador": r["Nombre"],
            "SPO2_block1": float(r["SPO2_1"]),
            "SPO2_last": float(r[f"SPO2_{last_j}"])
        })
paired_df = pd.DataFrame(paired)

x1 = paired_df["SPO2_block1"].to_numpy(float)
x2 = paired_df["SPO2_last"].to_numpy(float)
t = stats.ttest_rel(x1, x2)

plt.figure(figsize=(7,5))
parts = plt.violinplot([x1, x2], positions=[1,2], showmeans=True, showmedians=True, showextrema=True)
rng = np.random.default_rng(42)
j1 = rng.normal(1.0, 0.03, len(x1))
j2 = rng.normal(2.0, 0.03, len(x2))
plt.scatter(j1, x1, alpha=0.7)
plt.scatter(j2, x2, alpha=0.7)
for a,b,c,d in zip(j1,x1,j2,x2):
    plt.plot([a,c],[b,d], alpha=0.3)
plt.xticks([1,2], ["Block 1","Last block"])
plt.ylabel("SpO₂ (%)")
plt.title(f"SpO₂ paired: t({len(x1)-1})={t.statistic:.2f}, p={t.pvalue:.3f}")
plt.tight_layout()
plt.savefig(OUT_FIG/"violin_paired_spo2.png", dpi=200)
plt.close()

# ---------------------------------------------------
# 2) GPS: RESUMEN RÁPIDO + GRÁFICO m/min vs MAI/min
# ---------------------------------------------------
# Lectura robusta para CSV
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
    # si existieran en tu CSV:
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

# Guardar resumen por jugador (modo “rápido”: solo columnas disponibles)
keep = [c for c in [
    "name","minutes","distance_m","m_per_min","hsr_m","sprint_m","mai_m",
    "dist_per90","hsr_per90","sprint_per90","mai_per90","mai_per_min",
    "Position","Line"
] if c in df.columns]
summary = df[keep].copy()

# Asegurar que existan Position/Line aunque no vengan en el CSV  <<<<< FIX CLAVE
for col in ("Position","Line"):
    if col not in summary.columns:
        summary[col] = "Unknown"

summary.to_csv(OUT_TAB/"gps_summary_by_player.csv", index=False)

# --- Filtrado y alias antes del scatter ---
# Limitar mean (hipoxia) a jugadores del CSV del partido
gps_names_key = set(summary["name"].apply(canonical_key))
mean_filtered = mean[mean["Nombre"].apply(canonical_key).isin(gps_names_key)].copy()

# Excluir filas no-jugador (ej. "Promedio")
to_exclude = {"promedio", "equipo", "avg", "average"}
mean_filtered = mean_filtered[~mean_filtered["Nombre"].apply(lambda x: canonical_key(x) in to_exclude)]

# Aplica alias a ambos datasets
summary["name_key"] = summary["name"].apply(canonical_key).apply(apply_alias)
mean_filtered["name_key"] = mean_filtered["Nombre"].apply(canonical_key).apply(apply_alias)

# ---------------------------------------------------
# Gráficos m/min vs MAI/min: básico, con nombres y por línea
# ---------------------------------------------------
# Asegurar columnas base
if {"name","m_per_min","mai_per_min"}.issubset(summary.columns):
    # Construir columnas garantizando Position/Line  <<<<< FIX CLAVE
    cols = ["name","m_per_min","mai_per_min","Position","Line"]
    plot_df = summary[cols].copy()

    # Normalizar Line/Position
    for col in ["Position", "Line"]:
        plot_df[col] = plot_df[col].fillna("").astype(str).str.strip()
        plot_df.loc[plot_df[col] == "", col] = "Unknown"

    # ---------- (1) Scatter básico ----------
    plot_df_basic = plot_df.dropna(subset=["m_per_min","mai_per_min"])
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
        plt.xlabel("m/min"); plt.ylabel("MAI/min"); plt.title("Relación m/min vs MAI/min (partido)")
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT_FIG/"m_permin_vs_mai_permin_basico.png", dpi=200); plt.close()

    # ---------- (2) Scatter con nombres ----------
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
        plt.xlabel("m/min"); plt.ylabel("MAI/min"); plt.title("Relación m/min vs MAI/min (con nombres)")
        if len(plot_df_basic) >= 2: plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_FIG/"m_permin_vs_mai_permin_labels.png", dpi=200); plt.close()

    # ---------- (3) Scatter por línea (DEF/MID/FWD/Unknown) ----------
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

    # (Opcional) chequear quién quedó como Unknown
    unknowns = plot_df.loc[plot_df["Line"] == "Unknown", "name"].unique().tolist()
    if unknowns:
        print("⚠️ Jugadores con línea 'Unknown' en el gráfico:", ", ".join(unknowns))
    else:
        print("✅ Ningún jugador quedó con línea 'Unknown' en el gráfico.")
else:
    print("Aviso: summary no tiene columnas necesarias: name, m_per_min, mai_per_min.")

# ---------------------------------------------------
# Scatter m/min vs SpO₂ (última sesión) coloreado por línea
# ---------------------------------------------------
# Última SpO2 disponible por jugador (de mean_filtrado)
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

# Merge robusto: SpO₂_last ←→ GPS (m/min)
merged = pd.merge(
    spo2_last_df,
    summary[["name_key","name","m_per_min"]],
    on="name_key", how="left"
)

# (Opcional) Positions
pos_path = Path("data/positions.csv")
if pos_path.exists():
    pos_df = pd.read_csv(pos_path)
    if "Name" in pos_df.columns and "name" not in pos_df.columns:
        pos_df = pos_df.rename(columns={"Name":"name"})
    for col in ("Position","Line"):
        if col not in pos_df.columns:
            pos_df[col] = ""
    pos_df["name_key"] = pos_df["name"].apply(canonical_key).apply(apply_alias)
    merged = pd.merge(merged, pos_df[["name_key","Position","Line"]], on="name_key", how="left")
else:
    merged["Position"] = ""
    merged["Line"] = ""

# Normalizar Line y graficar
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
