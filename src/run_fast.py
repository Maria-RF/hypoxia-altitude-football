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

# ---------------------------------------------------
# Scatter m/min vs SpO₂ (última sesión) coloreado por línea (DEF/MID/FWD)
# Matching robusto a tildes/ñ/espacios
# ---------------------------------------------------
import re, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

def canonical_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 1) Última SpO₂ disponible por jugador (de 'mean' ya cargado)
spo2_cols_all = [c for c in mean.columns if str(c).startswith("SPO2_")]
last_records = []
for _, row in mean.iterrows():
    valid = [(c, row[c]) for c in spo2_cols_all if pd.notna(row.get(c))]
    if valid:
        last_col, last_val = valid[-1]
        last_records.append({"name_excel": row["Nombre"], "name_key": canonical_key(row["Nombre"]), "SPO2_last": float(last_val)})
spo2_last_df = pd.DataFrame(last_records)

# 2) GPS summary con clave
sum_use = summary.copy()
if "name" not in sum_use.columns:
    name_like = [c for c in sum_use.columns if c.lower() in ("name","nombre","jugador","player")]
    if name_like:
        sum_use = sum_use.rename(columns={name_like[0]:"name"})
sum_use["name_key"] = sum_use["name"].apply(canonical_key)

# 3) Cargar posiciones y crear clave
pos_path = Path("data/positions.csv")
pos_df = None
if pos_path.exists():
    pos_df = pd.read_csv(pos_path)
    if "Name" in pos_df.columns and "name" not in pos_df.columns:
        pos_df = pos_df.rename(columns={"Name":"name"})
    for col in ("Position","Line"):
        if col not in pos_df.columns:
            pos_df[col] = ""
    pos_df["name_key"] = pos_df["name"].apply(canonical_key)

# 4) Merge robusto: SpO₂_last ←→ GPS (m/min) ←→ Positions (Line)
merged = pd.merge(
    spo2_last_df[["name_key","name_excel","SPO2_last"]],
    sum_use[["name_key","name","m_per_min"]],
    on="name_key", how="left"
)

if pos_df is not None:
    merged = pd.merge(
        merged,
        pos_df[["name_key","Position","Line"]],
        on="name_key", how="left"
    )
else:
    merged["Position"] = ""
    merged["Line"] = ""

# Normalizar Line: NaN/"" -> "Unknown"
merged["Line"] = merged["Line"].fillna("").astype(str).str.strip()
merged.loc[merged["Line"]=="","Line"] = "Unknown"

# 5) Preparar datos completos para graficar
ok = merged.dropna(subset=["m_per_min","SPO2_last"]).copy()

# Mensajes de diagnóstico útiles en consola
faltan_gps = merged[merged["m_per_min"].isna()]["name_excel"].unique().tolist()
if faltan_gps:
    print("⚠️ Sin m/min en CSV para:", ", ".join(faltan_gps))
unknowns = ok.loc[ok["Line"]=="Unknown","name"].fillna(ok.loc[ok["Line"]=="Unknown","name_excel"]).unique().tolist()
if unknowns:
    print("⚠️ Jugadores sin línea conocida (Unknown):", ", ".join(map(str, unknowns)))

# 6) Scatter por línea de juego
if len(ok) >= 2:
    plt.figure(figsize=(8,6))
    for ln in ok["Line"].unique():
        sub = ok[ok["Line"] == ln]
        plt.scatter(sub["m_per_min"], sub["SPO2_last"], alpha=0.85, label=str(ln))
    # Línea OLS global + Pearson
    x = ok["m_per_min"].to_numpy(float)
    y = ok["SPO2_last"].to_numpy(float)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    xx = np.linspace(x.min(), x.max(), 100)
    yy = slope*xx + intercept
    r, pval = pearsonr(x, y)
    plt.plot(xx, yy, linestyle="--", label=f"OLS global: y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel("m/min (partido)")
    plt.ylabel("SpO₂ última sesión (%)")
    plt.title(f"m/min vs SpO₂ última sesión por línea (n={len(ok)})\nPearson r={r:.2f}, p={pval:.3f}")
    plt.legend(title="Línea")
    plt.tight_layout()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG/"corr_mmin_spo2_last_by_line.png", dpi=200)
    plt.close()
    print("✅ Figura guardada:", OUT_FIG/"corr_mmin_spo2_last_by_line.png")
else:
    print("⚠️ Datos insuficientes para el scatter por línea (se necesitan ≥2 pares).")
