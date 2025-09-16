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
# 2.b) Gráficos m/min vs MAI/min: básico, con nombres y por posición
#     Requiere 'summary' con columnas: name, m_per_min, mai_per_min
#     y (opcional) positions.csv con columnas: Name,Position,Line
# ---------------------------------------------------

FIG_DIR = OUT_FIG  # ya definido arriba

# Asegurar que están las columnas necesarias
if {"name","m_per_min","mai_per_min"}.issubset(summary.columns):
    plot_df = summary[["name","m_per_min","mai_per_min"]].dropna().copy()

    # ---------- (1) Scatter básico ----------
    if len(plot_df) >= 2:
        plt.figure(figsize=(7,5))
        plt.scatter(plot_df["m_per_min"], plot_df["mai_per_min"], alpha=0.85)
        # Recta OLS
        x = plot_df["m_per_min"].to_numpy(float)
        y = plot_df["mai_per_min"].to_numpy(float)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        xx = np.linspace(x.min(), x.max(), 100)
        yy = slope*xx + intercept
        plt.plot(xx, yy, linestyle="--", label=f"OLS: y={slope:.2f}x+{intercept:.2f}")
        plt.xlabel("m/min")
        plt.ylabel("MAI/min")
        plt.title("Relación m/min vs MAI/min (partido)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR/"m_permin_vs_mai_permin_basico.png", dpi=200)
        plt.close()
    else:
        print("Aviso: datos insuficientes para scatter básico (m/min vs MAI/min).")

    # ---------- (2) Scatter con nombres ----------
    if len(plot_df) >= 1:
        plt.figure(figsize=(8,6))
        plt.scatter(plot_df["m_per_min"], plot_df["mai_per_min"], alpha=0.7)
        # Etiquetas (nombre)
        for _, r in plot_df.iterrows():
            plt.annotate(str(r["name"]), (r["m_per_min"], r["mai_per_min"]), fontsize=8, alpha=0.85)
        # Recta OLS si hay >=2
        if len(plot_df) >= 2:
            x = plot_df["m_per_min"].to_numpy(float)
            y = plot_df["mai_per_min"].to_numpy(float)
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 100)
            yy = slope*xx + intercept
            plt.plot(xx, yy, linestyle="--", label=f"OLS: y={slope:.2f}x+{intercept:.2f}")
        plt.xlabel("m/min")
        plt.ylabel("MAI/min")
        plt.title("Relación m/min vs MAI/min (con nombres)")
        if len(plot_df) >= 2:
            plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR/"m_permin_vs_mai_permin_labels.png", dpi=200)
        plt.close()
    else:
        print("Aviso: datos insuficientes para scatter con nombres.")

    # ---------- (3) Scatter por posición (Line) ----------
    # Carga mapping posiciones si existe
    POS_FILE = Path("data/positions.csv")
    if POS_FILE.exists():
        pos = pd.read_csv(POS_FILE)
        # Normalizar clave de unión: Name -> name
        if "Name" in pos.columns:
            pos = pos.rename(columns={"Name": "name"})
        # Merge
        plot_pos = plot_df.merge(pos[["name","Position","Line"]], on="name", how="left")
        # Categorías de color (DEF/MID/FWD/Unknown)
        def label_line(x):
            return x if pd.notna(x) else "Unknown"
        plot_pos["Line"] = plot_pos["Line"].apply(label_line)

        # Colorear por línea
        lines = plot_pos["Line"].unique().tolist()
        plt.figure(figsize=(8,6))
        for ln in lines:
            sub = plot_pos[plot_pos["Line"] == ln]
            plt.scatter(sub["m_per_min"], sub["mai_per_min"], alpha=0.85, label=str(ln))
        # Recta OLS global
        if len(plot_pos) >= 2:
            x = plot_pos["m_per_min"].to_numpy(float)
            y = plot_pos["mai_per_min"].to_numpy(float)
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 100)
            yy = slope*xx + intercept
            plt.plot(xx, yy, linestyle="--", label=f"OLS global: y={slope:.2f}x+{intercept:.2f}")
        plt.xlabel("m/min")
        plt.ylabel("MAI/min")
        plt.title("Relación m/min vs MAI/min por línea (DEF/MID/FWD)")
        plt.legend(title="Línea")
        plt.tight_layout()
        plt.savefig(FIG_DIR/"m_permin_vs_mai_permin_por_linea.png", dpi=200)
        plt.close()

        # (Opcional) Etiquetas por posición si quieres otra versión:
        # Descomenta para un scatter con nombres y color por línea
        """
        plt.figure(figsize=(9,6))
        color_map = {ln: None for ln in lines}  # usa colores por defecto
        for ln in lines:
            sub = plot_pos[plot_pos['Line'] == ln]
            plt.scatter(sub["m_per_min"], sub["mai_per_min"], alpha=0.8, label=str(ln))
            for _, r in sub.iterrows():
                plt.annotate(str(r["name"]), (r["m_per_min"], r["mai_per_min"]), fontsize=8, alpha=0.8)
        plt.xlabel("m/min"); plt.ylabel("MAI/min")
        plt.title("m/min vs MAI/min (nombres + color por línea)")
        plt.legend(title="Línea")
        plt.tight_layout()
        plt.savefig(FIG_DIR/"m_permin_vs_mai_permin_linea_labels.png", dpi=200)
        plt.close()
        """
    else:
        print("Nota: no se encontró data/positions.csv; se omite el gráfico por posición.")
else:
    print("Aviso: summary no tiene columnas necesarias: name, m_per_min, mai_per_min.")

