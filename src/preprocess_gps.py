import pandas as pd
import numpy as np

def read_gps(csv_path):
    # Try ; then , then sniff
    try:
        df = pd.read_csv(csv_path, sep=';', engine='python')
    except Exception:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            import csv
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(2048)
                dialect = csv.Sniffer().sniff(sample)
                delim = dialect.delimiter
            df = pd.read_csv(csv_path, sep=delim, engine='python')
    return df

def comma_to_float(s):
    if isinstance(s, str):
        s = s.replace('.', '').replace(',', '.')
        try: return float(s)
        except: return np.nan
    return s

def clean_gps(df):
    df = df.copy()
    # Normalize likely columns
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "minutos" in cl or "minutes" in cl:
            rename[c] = "minutes"
        if "distance" in cl or "dist" in cl or "metros" in cl:
            rename[c] = "distance_m"
        if "m/min" in cl or "mts/min" in cl:
            rename[c] = "m_per_min"
        if "mai" in cl and "/min" in cl:
            rename[c] = "mai_per_min"
        if "mai" in cl and "/min" not in cl and "%" not in cl:
            rename[c] = "mai_m"
        if "sprint" in cl and "m" in cl:
            rename[c] = "sprint_m"
        if "hsr" in cl:
            rename[c] = "hsr_m"
        if "name" in cl:
            rename[c] = "Name"
    df = df.rename(columns=rename)
    # Convert comma decimals
    for c in ["minutes", "distance_m", "m_per_min", "mai_per_min", "mai_m", "sprint_m", "hsr_m"]:
        if c in df.columns:
            df[c] = df[c].apply(comma_to_float)
    # Derive per-90
    if "minutes" in df.columns:
        m = df["minutes"].replace(0, np.nan)
        for base, out in [("distance_m","dist_per90"),("hsr_m","hsr_per90"),("sprint_m","sprint_per90"),("mai_m","mai_per90")]:
            if base in df.columns:
                df[out] = df[base] * (90.0 / m)
    return df

def summarize_gps_by_player(df):
    if "Name" not in df.columns:
        # fall back to first column
        df = df.rename(columns={df.columns[0]:"Name"})
    def wavg(x, w):
        x, w = x.astype(float), w.astype(float)
        m = (~x.isna()) & (~w.isna()) & (w>0)
        return np.average(x[m], weights=w[m]) if m.sum() else np.nan
    grouped = []
    for name, g in df.groupby("Name"):
        minutes = g["minutes"] if "minutes" in g.columns else None
        row = {"Name": name}
        for c in ["distance_m","hsr_m","sprint_m","mai_m"]:
            if c in g.columns: row[c] = g[c].sum()
        for c in ["m_per_min","mai_per_min","dist_per90","hsr_per90","sprint_per90","mai_per90"]:
            if c in g.columns and minutes is not None:
                row[c+" (avg)"] = wavg(g[c], minutes)
        if minutes is not None:
            row["Minutes_total"] = g["minutes"].sum()
        grouped.append(row)
    return pd.DataFrame(grouped)
