#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
from pathlib import Path
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ======================================================
# ================= CONFIGURACIÓN ======================
# ======================================================

# Carpeta con accuracy.csv, precision.csv, recall.csv, specificity.csv, f1.csv, loss.csv
INPUT_DIR = Path("/home/cristiancrr/Documents/unfermented")
# Carpeta de salida
OUT_DIR   = Path("/home/cristiancrr/Documents/unfermented_plots")

# Modelos OBLIGATORIOS (todos deben aparecer en cada métrica)
REQUIRED_MODELS = [
    "vit_large",
    "convnext_xlarge",
    "vit_base",
    "mobilenetv3_large",
    "efficientnet_b5",
    # agrega más si los necesitas:
    # "vgg19", "convnext_xxlarge", "vit_base", "vgg16", "vgg13",
]

# Métricas a graficar (un CSV por cada una en INPUT_DIR)
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "loss"]

# Para la métrica 'loss' aceptamos estos alias por modelo
METRIC_ALIASES = {
    "loss": ["loss", "train_loss", "val_loss"],
    # El resto usan nombre exacto
    "accuracy": ["accuracy"],
    "precision": ["precision"],
    "recall": ["recall"],
    "specificity": ["specificity"],
    "f1": ["f1"],
}

# Estilo/grabado
DPI = 220
SAVE_PDF = False  # pon True si también quieres PDF vectorial

# ======================================================
# ================== ESTILO GLOBAL =====================
# ======================================================
plt.rcParams.update({
    "font.family": "DejaVu Serif",   # legible tipo paper
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.linewidth": 1.1,
    "lines.linewidth": 2.0,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ======================================================
# ==================== UTILIDADES ======================
# ======================================================
def slugify(text: str) -> str:
    text = (unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore").decode("ascii"))
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text) or "metric"

def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception:
        for sep in [";", "\t", "|"]:
            try:
                return pd.read_csv(csv_path, sep=sep)
            except Exception:
                continue
        raise

def find_x_axis(df: pd.DataFrame):
    # Usa epoch/step/iteration si existe; si no, índice 1..N
    for cand in ["epoch", "step", "iteration"]:
        for col in df.columns:
            if str(col).strip().lower() == cand:
                return col, pd.to_numeric(df[col], errors="coerce")
    idx = pd.Series(range(1, len(df) + 1), index=df.index, dtype="float64")
    return "index", idx

def ensure_unit_interval(y: pd.Series) -> bool:
    y_clean = pd.to_numeric(y, errors="coerce").dropna()
    return (not y_clean.empty) and (y_clean.min() >= -0.05) and (y_clean.max() <= 1.05)

def set_wandb_like_ylim(ax, series_list):
    """
    Ajusta los límites Y al estilo W&B: arranca cerca del mínimo observado,
    con un padding proporcional. Si parece [0,1], recorta a ese intervalo.
    """
    y_all = []
    for s in series_list:
        y_all.append(pd.to_numeric(s, errors="coerce"))
    y_cat = pd.concat(y_all, axis=0).dropna()
    if y_cat.empty:
        return
    ymin = float(y_cat.min())
    ymax = float(y_cat.max())
    rng = max(1e-9, ymax - ymin)
    pad = max(0.02, 0.06 * rng)  # 6% del rango, mínimo 0.02
    lower = ymin - pad
    upper = ymax + pad
    if ensure_unit_interval(y_cat):
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    if upper - lower < 1e-6:  # evitar límites iguales
        upper = lower + 0.05
    ax.set_ylim(lower, upper)

def norm(s: str) -> str:
    return str(s).strip().lower()

def build_norm_map(columns) -> dict:
    """mapa col_normalizada -> nombre_original"""
    m = {}
    for c in columns:
        m[norm(c)] = c
    return m

def pick_required_triplet(model: str, metric: str, norm_map: dict):
    """
    Retorna (main_col, min_col, max_col) **con los nombres originales**,
    eligiendo el primer alias disponible para 'metric'.
    Lanza ValueError detallando las faltantes si no encuentra un alias completo.
    """
    aliases = METRIC_ALIASES.get(metric, [metric])
    tried = []
    for alias in aliases:
        base = f"{model} - {alias}"
        main_key = norm(base)
        min_key  = norm(f"{base}__MIN")
        max_key  = norm(f"{base}__MAX")
        ok_main = main_key in norm_map
        ok_min  = min_key  in norm_map
        ok_max  = max_key  in norm_map
        if ok_main and ok_min and ok_max:
            return norm_map[main_key], norm_map[min_key], norm_map[max_key]
        tried.append((base, ok_main, ok_min, ok_max))
    # Si ningún alias cumplió, construir mensaje claro
    missing_details = []
    for base, ok_main, ok_min, ok_max in tried:
        if not ok_main: missing_details.append(f"'{base}'")
        if not ok_min:  missing_details.append(f"'{base}__MIN'")
        if not ok_max:  missing_details.append(f"'{base}__MAX'")
    raise ValueError(", ".join(missing_details))

# ======================================================
# ======================= MAIN =========================
# ======================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📂 INPUT_DIR: {INPUT_DIR}")
    print(f"💾 OUT_DIR:   {OUT_DIR}")
    print(f"🧠 MODELOS (obligatorios): {REQUIRED_MODELS}")
    print(f"🧪 MÉTRICAS: {METRICS}")

    for metric in METRICS:
        csv_path = INPUT_DIR / f"{metric}.csv"
        if not csv_path.exists():
            print(f"❌ Falta el archivo: {csv_path}")
            sys.exit(1)

        # Leer CSV de la métrica
        try:
            df = read_csv_safely(csv_path)
        except Exception as e:
            print(f"❌ No se pudo leer {csv_path.name}: {e}")
            sys.exit(1)

        # Mapa normalizado -> nombre original (para tolerar mayúsculas/espacios)
        norm_map = build_norm_map(df.columns)

        # Verificar columnas obligatorias (main/MIN/MAX) por modelo con alias
        missing_report = []
        model_cols = {}  # model -> (main,min,max)
        for m in REQUIRED_MODELS:
            try:
                model_cols[m] = pick_required_triplet(m, metric, norm_map)
            except ValueError as ve:
                missing_report.append(f"[{csv_path.name}] {m}: faltan {ve}")

        if missing_report:
            print("❌ Columnas obligatorias faltantes:")
            for line in missing_report:
                print(f"   - {line}")
            sys.exit(1)

        # Eje X
        x_name, x_vals = find_x_axis(df)

        # Figura: todos los modelos en una (con banda MIN–MAX)
        fig = plt.figure(figsize=(11.2, 6.4))
        ax = plt.gca()
        y_for_limits = []

        for m in REQUIRED_MODELS:
            main_c, min_c, max_c = model_cols[m]

            y = pd.to_numeric(df[main_c], errors="coerce")
            ymin = pd.to_numeric(df[min_c], errors="coerce")
            ymax = pd.to_numeric(df[max_c], errors="coerce")

            # Validez simultánea en X, Y, Ymin, Ymax
            valid = x_vals.notna() & y.notna() & ymin.notna() & ymax.notna()
            x = x_vals[valid]
            yv = y[valid]
            yvmin = ymin[valid]
            yvmax = ymax[valid]

            if x.empty or yv.empty:
                print(f"❌ {csv_path.name}: datos vacíos tras limpieza para {m}.")
                sys.exit(1)

            ax.plot(x, yv, label=f"{m} (best={float(yv.max()):.4f})")
            ax.fill_between(x, yvmin, yvmax, alpha=0.18)
            y_for_limits.append(yv)

        # Estética tipo paper + ejes estilo W&B
        ax.set_title(metric.upper())
        ax.set_xlabel(x_name)
        ax.set_ylabel(metric.capitalize() if metric != "loss" else "Loss")
        ax.grid(True, which="major", alpha=0.35)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        set_wandb_like_ylim(ax, y_for_limits)

        # Leyenda externa superior
        n_models = len(REQUIRED_MODELS)
        ncols = 3 if n_models <= 9 else 4
        ax.legend(frameon=False, ncol=ncols, bbox_to_anchor=(0.5, 1.18), loc="upper center")

        fig.tight_layout()
        out_png = OUT_DIR / f"{slugify(metric)}_todos_los_modelos.png"
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        if SAVE_PDF:
            out_pdf = OUT_DIR / f"{slugify(metric)}_todos_los_modelos.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"✅ {metric}: guardado {out_png}")

    print("🎉 Listo. Todas las figuras han sido generadas en:", OUT_DIR)


if __name__ == "__main__":
    main()
