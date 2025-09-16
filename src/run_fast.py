# scripts/run_fast.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend no interactivo (más rápido/estable)
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------
# RUTAS (ajusta si es necesario)
# ----------------------------
EXCEL = "data/DataBase_Futbol_2025v.xlsx"
CSV   = "data/export_15-09-25.csv"

OUT_FIG = Path("outputs/figures")
OUT_TAB = Path("outputs/tables")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

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
# Lectura robusta para CSV con ; y coma decimal (formato LatAm)
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
    "SPRINT>25km/h":"sprint_m"
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
    "dist_per90","hsr_per90","sprint_per90","mai_per90","mai_per_min"
] if c in df.columns]
summary = df[keep].copy()
summary.to_csv(OUT_TAB/"gps_summary_by_player.csv", index=False)

# --------- NUEVO: Gráfico m/min vs MAI/min (partido) ----------
if ("m_per_min" in df.columns) and ("mai_per_min" in df.columns):
    # Quitar NaN e infinitos
    plot_df = df[["name","m_per_min","mai_per_min"]].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(7,5))
    plt.scatter(plot_df["m_per_min"], plot_df["mai_per_min"], alpha=0.8)
    # Ajuste lineal simple (opcional)
    if len(plot_df) >= 2:
        x = plot_df["m_per_min"].to_numpy(float)
        y = plot_df["mai_per_min"].to_numpy(float)
        # recta de regresión
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope*x_line + intercept
        plt.plot(x_line, y_line, linestyle="--", label=f"OLS: y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel("m/min")
    plt.ylabel("MAI/min")
    plt.title("Relación m/min vs MAI/min (partido)")
    if len(plot_df) >= 2:
        plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG/"m_permin_vs_mai_permin.png", dpi=200)
    plt.close()
else:
    print("Aviso: columnas necesarias no disponibles para el gráfico m/min vs MAI/min.")

print("FAST mode listo ✅")
print(" - outputs/figures/violin_paired_spo2.png")
print(" - outputs/figures/m_permin_vs_mai_permin.png (si había columnas)")
print(" - outputs/tables/gps_summary_by_player.csv")
