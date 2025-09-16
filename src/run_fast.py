# scripts/run_fast.py
from pathlib import Path
import pandas as pd
import numpy as np

EXCEL = "data/DataBase_Futbol_2025v.xlsx"
CSV   = "data/export_15-09-25.csv"

OUT_FIG = Path("outputs/figures"); OUT_TAB = Path("outputs/tables")
OUT_FIG.mkdir(parents=True, exist_ok=True); OUT_TAB.mkdir(parents=True, exist_ok=True)

# 1) Hipoxia: paired 1 vs last (>=4 sesiones)
mean = pd.read_excel(EXCEL, sheet_name="Mean Datos por Sesion")
res  = pd.read_excel(EXCEL, sheet_name="Resumen")
mean = mean.merge(res[["Nombre","Total de sesiones (n)"]], on="Nombre", how="left")
mean4 = mean[mean["Total de sesiones (n)"]>=4].copy()

paired = []
for _, r in mean4.iterrows():
    if pd.notna(r.get("SPO2_1")):
        last_j = max([j for j in range(1,10) if f"SPO2_{j}" in mean4.columns and pd.notna(r.get(f"SPO2_{j}"))])
        paired.append({"Jugador": r["Nombre"], "SPO2_block1": r["SPO2_1"], "SPO2_last": r[f"SPO2_{last_j}"]})
paired_df = pd.DataFrame(paired)

# violin
import matplotlib.pyplot as plt
from scipy import stats
x1 = paired_df["SPO2_block1"].to_numpy(float)
x2 = paired_df["SPO2_last"].to_numpy(float)
t = stats.ttest_rel(x1, x2)

plt.figure(figsize=(7,5))
parts = plt.violinplot([x1, x2], positions=[1,2], showmeans=True, showmedians=True, showextrema=True)
rng = np.random.default_rng(42)
j1 = rng.normal(1.0, 0.03, len(x1)); j2 = rng.normal(2.0, 0.03, len(x2))
plt.scatter(j1, x1, alpha=0.7); plt.scatter(j2, x2, alpha=0.7)
for a,b,c,d in zip(j1,x1,j2,x2): plt.plot([a,c],[b,d], alpha=0.3)
plt.xticks([1,2], ["Block 1","Last block"])
plt.ylabel("SpO₂ (%)")
plt.title(f"SpO₂ paired: t({len(x1)-1})={t.statistic:.2f}, p={t.pvalue:.3f}")
plt.tight_layout()
plt.savefig(OUT_FIG/"violin_paired_spo2.png", dpi=200)
plt.close()

# 2) GPS: cargar rápido y resumir por jugador
df = pd.read_csv(CSV, sep=";", decimal=",", engine="python", encoding="utf-8-sig")
# renombres mínimos
rename = {
    "Name":"name","Minutos":"minutes","Total Distance (m)":"distance_m",
    "Mts/min":"m_per_min","MAInt >20km/h":"mai_m","AInt >15km/h":"hsr_m",
    "SPRINT>25km/h":"sprint_m"
}
df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
# per90
if "minutes" in df.columns:
    m = df["minutes"].replace(0, np.nan)
    for base, out in [("distance_m","dist_per90"),("hsr_m","hsr_per90"),
                      ("sprint_m","sprint_per90"),("mai_m","mai_per90")]:
        if base in df.columns: df[out] = df[base]*(90.0/m)
# resumen simple
keep = [c for c in ["name","minutes","distance_m","m_per_min","hsr_m","sprint_m","mai_m",
                    "dist_per90","hsr_per90","sprint_per90","mai_per90"] if c in df.columns]
summary = df[keep].copy()
summary.to_csv(OUT_TAB/"gps_summary_by_player.csv", index=False)

print("FAST mode listo ✅ → outputs/figures/violin_paired_spo2.png y outputs/tables/gps_summary_by_player.csv")
