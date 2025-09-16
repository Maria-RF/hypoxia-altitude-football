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
# Merge con posiciones robusto a tildes y ñ
# ---------------------------------------------------
from pathlib import Path
import re
import unicodedata
import pandas as pd

POS_FILE = Path("data/positions.csv")

def canonical_key(s: str) -> str:
    """
    Normaliza nombres para hacer match robusto:
    - casefold (minúsculas robustas)
    - NFKD + elimina diacríticos (tildes); 'ñ' -> 'n'
    - elimina signos/puntuación
    - colapsa y recorta espacios
    """
    if s is None:
        return ""
    # a string
    s = str(s)
    # minúsculas robustas
    s = s.casefold()
    # normaliza unicode y elimina diacríticos (tildes)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # opcional: mapear caracteres similares (ñ → n ya queda como 'n' tras NFKD)
    # limpiar puntuación y no-alfanumérico (conserva espacios)
    s = re.sub(r"[^\w\s]", " ", s)  # quita puntuación
    # colapsar espacios múltiples
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Asegura la columna 'name' en summary (si no existe, intenta inferir)
if "name" not in summary.columns:
    # intenta detectar columna de nombre
    name_like = [c for c in summary.columns if c.lower() in ("name","nombre","jugador","player")]
    if not name_like:
        name_like = [summary.columns[0]]
    summary = summary.rename(columns={name_like[0]: "name"})

# Crea clave canónica en summary
summary["name_key"] = summary["name"].apply(canonical_key)

if POS_FILE.exists():
    pos = pd.read_csv(POS_FILE)

    # normaliza encabezado y crea clave canónica también en positions
    # acepta 'Name' o 'name'
    if "Name" in pos.columns and "name" not in pos.columns:
        pos = pos.rename(columns={"Name": "name"})
    # asegurar columnas Position y Line (si no están, créalas vacías)
    if "Position" not in pos.columns:
        pos["Position"] = ""
    if "Line" not in pos.columns:
        pos["Line"] = ""
    pos["name_key"] = pos["name"].apply(canonical_key)

    # merge por clave canónica (left join para no perder nadie del resumen GPS)
    merged = summary.merge(
        pos[["name_key","Position","Line","name"]].rename(columns={"name":"name_positions"}),
        on="name_key",
        how="left",
        validate="m:1"
    )

    # completa Position/Line con vacío si falta
    merged["Position"] = merged["Position"].fillna("")
    merged["Line"] = merged["Line"].fillna("")

    # Reporte de no matcheados
    no_match = merged[merged["Position"].eq("") & merged["Line"].eq("")]["name"].unique().tolist()
    if no_match:
        print("⚠️ Sin match en positions.csv para:", ", ".join(no_match))
    else:
        print("✅ Todos los jugadores del resumen GPS tienen posición/linea (o se completó).")

    # Guarda tabla final
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_TAB / "gps_summary_with_positions.csv", index=False)

    # (Opcional) reemplaza summary por merged para que los gráficos posteriores ya vean Position/Line
    summary = merged

else:
    print("ℹ️ No se encontró data/positions.csv; se continúa sin posiciones.")

# ---------------------------------------------------
# Gráficos m/min vs MAI/min: básico, con nombres y por línea (usando summary ya fusionado)
# ---------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Asegurar columnas base
if {"name","m_per_min","mai_per_min"}.issubset(summary.columns):
    plot_df = summary[["name","m_per_min","mai_per_min","Position","Line"]].copy()

    # Normalizar Line/Position: NaN o "" -> "Unknown"
    for col in ["Position", "Line"]:
        if col not in plot_df.columns:
            plot_df[col] = "Unknown"
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
# Correlación m/min (partido) vs SpO₂ última sesión (hipoxia)
# ---------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- 1) Última sesión SpO₂ por jugador
# mean4 es la tabla con todas las sesiones por jugador (SPO2_1 ... SPO2_n)
spo2_cols = [c for c in mean4.columns if c.startswith("SPO2_")]
spo2_last = []
for _, row in mean4.iterrows():
    # última sesión con dato válido
    valid_cols = [c for c in spo2_cols if not pd.isna(row[c])]
    if valid_cols:
        last_col = valid_cols[-1]
        spo2_last.append({"name": row["Nombre"], "SPO2_last": row[last_col]})
spo2_last_df = pd.DataFrame(spo2_last)

# --- 2) Merge con summary para tener m/min
if "name" not in summary.columns:
    name_like = [c for c in summary.columns if c.lower() in ("name","nombre","jugador")]
    if name_like:
        summary = summary.rename(columns={name_like[0]:"name"})

merged_corr = spo2_last_df.merge(
    summary[["name","m_per_min"]], 
    on="name", 
    how="left"
)

# --- 3) Scatter + OLS + correlación
merged_corr = merged_corr.dropna(subset=["m_per_min","SPO2_last"])
if len(merged_corr) >= 2:
    x = merged_corr["m_per_min"].to_numpy(float)
    y = merged_corr["SPO2_last"].to_numpy(float)
    r, pval = pearsonr(x, y)

    plt.figure(figsize=(7,5))
    plt.scatter(x, y, alpha=0.85)
    # Línea OLS
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    xx = np.linspace(x.min(), x.max(), 100)
    yy = slope*xx + intercept
    plt.plot(xx, yy, "r--", label=f"OLS: y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel("m/min (partido)")
    plt.ylabel("SpO₂ última sesión (%)")
    plt.title(f"Correlación m/min vs SpO₂ última sesión\nr={r:.2f}, p={pval:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG/"corr_mmin_spo2_last.png", dpi=200)
    plt.close()

    print(f"✅ Gráfico de correlación guardado en outputs/figures/corr_mmin_spo2_last.png (r={r:.2f}, p={pval:.3f})")
else:
    print("⚠️ Datos insuficientes para correlación m/min vs SpO₂ última sesión.")

