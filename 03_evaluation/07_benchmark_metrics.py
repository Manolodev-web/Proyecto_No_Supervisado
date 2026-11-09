import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Configuración de Rutas y Carga de Datos
# ==============================================================================
current_dir = os.path.abspath(os.path.dirname(__file__))
base_dir = current_dir
while base_dir and not os.path.exists(os.path.join(base_dir, "00_data")):
  parent = os.path.dirname(base_dir)
  if parent == base_dir:
    break
  base_dir = parent

PROCESSED_DIR = os.path.join(base_dir, "00_data", "processed")
REPORTS_DIR = os.path.join(base_dir, "05_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

df_feats = pd.read_csv(os.path.join(PROCESSED_DIR, "features_instruments.csv"))
df_km = pd.read_csv(os.path.join(PROCESSED_DIR, "results_kmeans.csv"))
df_hier = pd.read_csv(os.path.join(PROCESSED_DIR, "results_hierarchical.csv"))
df_dens = pd.read_csv(os.path.join(PROCESSED_DIR, "results_density.csv"))
df_gmm = pd.read_csv(os.path.join(PROCESSED_DIR, "results_gmm.csv"))

feature_cols = [
    c for c in df_feats.columns if c not in ["file_name", "true_instrument"]
]
X_scaled = StandardScaler().fit_transform(df_feats[feature_cols].values)
X_eval = PCA(n_components=15, random_state=42).fit_transform(X_scaled)
true_labels = df_feats["true_instrument"].values

# ==============================================================================
# 2. Evaluación Unificada de Métricas
# ==============================================================================
modelos_labels = {
    "K-Means (k=4)": df_km["cluster_kmeans"].values,
    "Jerárquico Ward (k=4)": df_hier["cluster_hierarchical"].values,
    "DBSCAN": df_dens["cluster_dbscan"].values,
    "HDBSCAN": df_dens["cluster_hdbscan"].values,
    "GMM (k=4)": df_gmm["cluster_asignado"].values,
}

resultados = []

for nombre, labs in modelos_labels.items():
  mask = labs != -1
  n_clusters = len(set(labs[mask]))
  n_ruido = int(np.sum(labs == -1))
  pct_ruido = round((n_ruido / len(labs)) * 100, 2)

  if n_clusters > 1 and np.sum(mask) > n_clusters:
    sil = round(float(silhouette_score(X_eval[mask], labs[mask])), 4)
    ch = round(float(calinski_harabasz_score(X_eval[mask], labs[mask])), 2)
    db = round(float(davies_bouldin_score(X_eval[mask], labs[mask])), 4)
    ari = round(float(adjusted_rand_score(true_labels[mask], labs[mask])), 4)
  else:
    sil, ch, db, ari = 0.0, 0.0, 0.0, 0.0

  resultados.append({
      "Modelo": nombre,
      "N° Clusters": n_clusters,
      "Ruido": f"{n_ruido} ({pct_ruido}%)",
      "Silhouette (↑)": sil,
      "Calinski-H (↑)": ch,
      "Davies-Bouldin (↓)": db,
      "ARI (↑)": ari,
  })

df_benchmark = pd.DataFrame(resultados)

print("\n" + "=" * 85)
print("     CUADRO COMPARATIVO CONSOLIDADO DE APRENDIZAJE NO SUPERVISADO")
print("=" * 85)
print(df_benchmark.to_string(index=False))
print("=" * 85)

# Guardar tabla en CSV para anexos del informe
benchmark_csv = os.path.join(REPORTS_DIR, "final_benchmark_metrics.csv")
df_benchmark.to_csv(benchmark_csv, index=False)

# ==============================================================================
# 3. Gráfico Comparativo Multimétrica
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

nombres = [r["Modelo"] for r in resultados]
sil_vals = [r["Silhouette (↑)"] for r in resultados]
ari_vals = [r["ARI (↑)"] for r in resultados]

colores = ["#3b82f6", "#10b981", "#f59e0b", "#06b6d4", "#8b5cf6"]

ax1.barh(nombres, sil_vals, color=colores, edgecolor="black", alpha=0.85)
ax1.set_title("Coeficiente de Silueta (Cohesión y Separación)", fontweight="bold")
ax1.set_xlabel("Puntuación (-1 a 1)")
ax1.grid(True, alpha=0.3)

ax2.barh(nombres, ari_vals, color=colores, edgecolor="black", alpha=0.85)
ax2.set_title(
    "Adjusted Rand Index - ARI (Correspondencia Externa)", fontweight="bold"
)
ax2.set_xlabel("Puntuación (0 a 1)")
ax2.grid(True, alpha=0.3)

plt.suptitle(
    "Benchmark Final de Modelos No Supervisados sobre Espacio CQT/HPSS",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()

chart_path = os.path.join(REPORTS_DIR, "benchmark_comparison_charts.png")
plt.savefig(chart_path, dpi=300)
plt.show()
print(f"Gráfico guardado en: {chart_path}")