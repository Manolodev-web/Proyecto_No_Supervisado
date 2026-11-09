import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Detección Dinámica de Rutas y Carga de Datos
# ==============================================================================
current_dir = os.path.abspath(os.path.dirname(__file__))
base_dir = current_dir
while base_dir and not os.path.exists(os.path.join(base_dir, "00_data")):
  parent = os.path.dirname(base_dir)
  if parent == base_dir:
    break
  base_dir = parent

DATA_PATH = os.path.join(
    base_dir, "00_data", "processed", "features_instruments.csv"
)
REPORTS_DIR = os.path.join(base_dir, "05_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
  raise FileNotFoundError(
      f"No se encontró {DATA_PATH}. Ejecuta 01_extract_features.py primero."
  )

print(f"Cargando dataset enriquecido desde: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

feature_cols = [
    c for c in df.columns if c not in ["file_name", "true_instrument"]
]
X_raw = df[feature_cols].values
true_labels = df["true_instrument"].values
file_names = df["file_name"].values

print(
    f"Matriz de datos cargada: {X_raw.shape[0]} muestras x {X_raw.shape[1]}"
    " características."
)

# Estandarización de las 71 variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ==============================================================================
# 2. Análisis de Varianza Explicada con PCA
# ==============================================================================
pca_full = PCA().fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_80 = int(np.argmax(cum_var >= 0.80) + 1)
n_comp_90 = int(np.argmax(cum_var >= 0.90) + 1)

print("\n--- ANÁLISIS DE REDUCCIÓN DE DIMENSIONALIDAD (PCA) ---")
print(f"Componentes para 80% de varianza explicada: {n_comp_80}")
print(f"Componentes para 90% de varianza explicada: {n_comp_90}")

# Reducción a las componentes esenciales para estabilizar la distancia euclidiana
n_comp_85 = max(n_comp_80, 8)
pca_dim = PCA(n_components=n_comp_85, random_state=42)
X_pca_reduced = pca_dim.fit_transform(X_scaled)

# ==============================================================================
# 3. Búsqueda de K Óptimo (Inercia / Codo y Silueta)
# ==============================================================================
k_range = range(2, 9)
inertias = []
silhouettes = []

print("\nEvaluando rango de clusters k in [2, 8]...")
for k in k_range:
  km = KMeans(n_clusters=k, random_state=42, n_init=10)
  labels_k = km.fit_predict(X_pca_reduced)
  inertias.append(km.inertia_)
  sil = silhouette_score(X_pca_reduced, labels_k)
  silhouettes.append(sil)
  print(f"  k={k} | Inercia: {km.inertia_:.1f} | Silueta: {sil:.4f}")

# Gráfico Codo y Silueta
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(k_range, inertias, marker="o", color="#2563eb", linewidth=2)
ax1.set_title("Método del Codo (Inercia)", fontweight="bold")
ax1.set_xlabel("Número de Clústeres (k)")
ax1.set_ylabel("Suma de Cuadrados Intra-Clúster")
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, silhouettes, marker="s", color="#10b981", linewidth=2)
ax2.set_title("Coeficiente de Silueta", fontweight="bold")
ax2.set_xlabel("Número de Clústeres (k)")
ax2.set_ylabel("Puntuación de Silueta")
ax2.grid(True, alpha=0.3)

plt.suptitle(
    "Diagnóstico de K Óptimo para Espacio Acústico CQT/HPSS", fontsize=14
)
plt.tight_layout()
elbow_path = os.path.join(REPORTS_DIR, "kmeans_elbow_silhouette.png")
plt.savefig(elbow_path, dpi=300)
plt.show()
print(f"Gráficas diagnósticas guardadas en: {elbow_path}")

# ==============================================================================
# 4. Ajuste K-Means Definitivo (k=4) y Métricas
# ==============================================================================
k_final = 4
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(X_pca_reduced)

sil = silhouette_score(X_pca_reduced, cluster_labels)
ch = calinski_harabasz_score(X_pca_reduced, cluster_labels)
db = davies_bouldin_score(X_pca_reduced, cluster_labels)
ari = adjusted_rand_score(true_labels, cluster_labels)

print("\n" + "=" * 65)
print(f"         RESULTADOS K-MEANS CON ESPACIO REDUCIDO (k={k_final})")
print("=" * 65)
print(f"Silhouette Score (↑):         {sil:.4f}")
print(f"Calinski-Harabasz Index (↑):  {ch:.2f}")
print(f"Davies-Bouldin Index (↓):     {db:.4f}")
print(f"Adjusted Rand Index - ARI (↑): {ari:.4f}")
print("=" * 65)

# ==============================================================================
# 5. Matriz de Contingencia (Cruce con Clases Reales)
# ==============================================================================
df_eval = pd.DataFrame(
    {"Instrumento_Real": true_labels, "Cluster_KMeans": cluster_labels}
)
contingency = pd.crosstab(
    df_eval["Instrumento_Real"],
    df_eval["Cluster_KMeans"],
    margins=True,
    margins_name="Total",
)

print("\n--- MATRIZ DE CONTINGENCIA (Validación Externa) ---")
print(contingency)
print("=" * 65)

# ==============================================================================
# 6. Visualización 2D: Clústeres Descubiertos vs Ground Truth
# ==============================================================================
pca_2d = PCA(n_components=2, random_state=42)
X_viz = pca_2d.fit_transform(X_scaled)
var_viz = pca_2d.explained_variance_ratio_

fig, (ax_km, ax_true) = plt.subplots(1, 2, figsize=(16, 6.5))

# Subplot 1: Asignaciones de K-Means
cmap_km = plt.colormaps["tab10"].resampled(k_final)
for k in range(k_final):
  mask = cluster_labels == k
  ax_km.scatter(
      X_viz[mask, 0],
      X_viz[mask, 1],
      color=cmap_km(k),
      alpha=0.55,
      s=30,
      label=f"Clúster {k}",
  )

# Centroides proyectados en 2D
centroids_scaled = scaler.inverse_transform(
    pca_dim.inverse_transform(kmeans_final.cluster_centers_)
)
centroids_2d = pca_2d.transform(scaler.transform(centroids_scaled))
ax_km.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    c="black",
    marker="X",
    s=180,
    edgecolor="white",
    linewidth=1.5,
    label="Centroides",
)
ax_km.set_title(
    f"Clustering K-Means (k={k_final})\nSilueta: {sil:.4f}",
    fontweight="bold",
)
ax_km.set_xlabel(f"PC1 ({var_viz[0]:.2%})")
ax_km.set_ylabel(f"PC2 ({var_viz[1]:.2%})")
ax_km.legend(loc="best", fontsize=9)
ax_km.grid(True, alpha=0.3)

# Subplot 2: Etiquetas Reales (Ground Truth)
inst_unicos = sorted(list(set(true_labels)))
cmap_gt = plt.colormaps["Set1"].resampled(len(inst_unicos))
for i, inst in enumerate(inst_unicos):
  mask = true_labels == inst
  ax_true.scatter(
      X_viz[mask, 0],
      X_viz[mask, 1],
      color=cmap_gt(i),
      alpha=0.55,
      s=30,
      label=inst,
  )

ax_true.set_title(
    f"Instrumentos Reales (Ground Truth)\nAdjusted Rand Index: {ari:.4f}",
    fontweight="bold",
)
ax_true.set_xlabel(f"PC1 ({var_viz[0]:.2%})")
ax_true.set_ylabel(f"PC2 ({var_viz[1]:.2%})")
ax_true.legend(loc="best", fontsize=9)
ax_true.grid(True, alpha=0.3)

plt.suptitle(
    "Comparativa Espacial: Agrupamiento No Supervisado vs Etiquetas Reales",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()

clusters_plot_path = os.path.join(REPORTS_DIR, "kmeans_pca_clusters.png")
plt.savefig(clusters_plot_path, dpi=300)
plt.show()
print(f"Gráfico de clústeres guardado en: {clusters_plot_path}")

# ==============================================================================
# 7. Exportar Resultados Tabulares
# ==============================================================================
df_results = pd.DataFrame({
    "file_name": file_names,
    "true_instrument": true_labels,
    "cluster_kmeans": cluster_labels,
})
output_res_path = os.path.join(
    base_dir, "00_data", "processed", "results_kmeans.csv"
)
df_results.to_csv(output_res_path, index=False)
print(f"Asignaciones guardadas en: {output_res_path}")