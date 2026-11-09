import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Configuración de Rutas y Datos
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "00_data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "05_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

CLEAN_CSV = os.path.join(PROCESSED_DIR, "features_instruments_clean.csv")
df_clean = pd.read_csv(CLEAN_CSV)
feat_cols = [c for c in df_clean.columns if c not in ["file_name", "true_instrument"]]

# Espacio latente unificado a 15D PCA
X_scaled = StandardScaler().fit_transform(df_clean[feat_cols].values)
pca_15d = PCA(n_components=15, random_state=42)
X_pca_15d = pca_15d.fit_transform(X_scaled)
y_true = df_clean["true_instrument"].values

# Cargar archivos de resultados de cada modelo
df_km = pd.read_csv(os.path.join(PROCESSED_DIR, "results_kmeans.csv"))
df_hier = pd.read_csv(os.path.join(PROCESSED_DIR, "results_hierarchical.csv"))
df_dens = pd.read_csv(os.path.join(PROCESSED_DIR, "results_density.csv"))
df_gmm = pd.read_csv(os.path.join(PROCESSED_DIR, "results_gmm.csv"))

modelos = {
    "K-Means (k=3)": df_km["cluster_kmeans"].values,
    "Jerárquico Ward (k=3)": df_hier["cluster_hierarchical"].values,
    "GMM (k=3)": df_gmm["cluster_asignado"].values,
    "DBSCAN (eps=3.4, min=12)": df_dens["cluster_dbscan"].values,
    "HDBSCAN (min_size=35)": df_dens["cluster_hdbscan"].values
}

# ==============================================================================
# 2. Evaluación Unificada de Métricas
# ==============================================================================
resultados = []

for nombre, labs in modelos.items():
    mask = labs != -1
    n_cl = len(set(labs[mask]))
    n_ruido = int(np.sum(~mask))
    pct_ruido = round((n_ruido / len(labs)) * 100, 2)

    if n_cl > 1:
        sil = round(float(silhouette_score(X_pca_15d[mask], labs[mask])), 4)
        ch = round(float(calinski_harabasz_score(X_pca_15d[mask], labs[mask])), 2)
        db = round(float(davies_bouldin_score(X_pca_15d[mask], labs[mask])), 4)
        ari = round(float(adjusted_rand_score(y_true[mask], labs[mask])), 4)
        nmi = round(float(normalized_mutual_info_score(y_true[mask], labs[mask])), 4)
    else:
        sil, ch, db, ari, nmi = 0.0, 0.0, 0.0, 0.0, 0.0

    resultados.append({
        "Modelo": nombre,
        "N° Clústeres": n_cl,
        "Ruido": f"{n_ruido} ({pct_ruido}%)",
        "Silueta (↑)": sil,
        "Calinski-H (↑)": ch,
        "Davies-Bouldin (↓)": db,
        "ARI (↑)": ari,
        "NMI (↑)": nmi
    })

df_benchmark = pd.DataFrame(resultados)

print("\n" + "=" * 95)
print("       CUADRO COMPARATIVO CONSOLIDADO DEFINITIVO (DATASET SANEADO - 15D PCA)")
print("=" * 95)
print(df_benchmark.to_string(index=False))
print("=" * 95)

# Exportar tabla única sin duplicidades
benchmark_csv = os.path.join(REPORTS_DIR, "final_benchmark_metrics.csv")
df_benchmark.to_csv(benchmark_csv, index=False)
print(f"\nTabla guardada en: {benchmark_csv}")

# ==============================================================================
# 3. Gráfica Multimétrica Comparativa
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

nombres = [r["Modelo"] for r in resultados]
sil_vals = [r["Silueta (↑)"] for r in resultados]
ari_vals = [r["ARI (↑)"] for r in resultados]
nmi_vals = [r["NMI (↑)"] for r in resultados]

colores = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#06b6d4"]

# Silueta
ax1.barh(nombres, sil_vals, color=colores, edgecolor="black", alpha=0.85)
ax1.set_title("Coeficiente de Silueta (Cohesión)", fontweight="bold")
ax1.set_xlabel("Puntuación (-1 a 1)")
ax1.grid(True, alpha=0.3)

# ARI
ax2.barh(nombres, ari_vals, color=colores, edgecolor="black", alpha=0.85)
ax2.set_title("Adjusted Rand Index - ARI", fontweight="bold")
ax2.set_xlabel("Puntuación (0 a 1)")
ax2.grid(True, alpha=0.3)

# NMI
ax3.barh(nombres, nmi_vals, color=colores, edgecolor="black", alpha=0.85)
ax3.set_title("Información Mutua Normalizada - NMI", fontweight="bold")
ax3.set_xlabel("Puntuación (0 a 1)")
ax3.grid(True, alpha=0.3)

plt.suptitle("Benchmark Final de Aprendizaje No Supervisado (k=3 Familias Acústicas)", fontsize=13, fontweight="bold")
plt.tight_layout()

chart_path = os.path.join(REPORTS_DIR, "benchmark_comparison_charts.png")
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Gráfico guardado en: {chart_path}")