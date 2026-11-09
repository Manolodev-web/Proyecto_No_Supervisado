import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Configuración de Rutas y Carga de Datos (71 Dimensiones)
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

df = pd.read_csv(DATA_PATH)
feature_cols = [
    c for c in df.columns if c not in ["file_name", "true_instrument"]
]
X_raw = df[feature_cols].values
true_labels = df["true_instrument"].values
file_names = df["file_name"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Reducción previa a 15 componentes (80% de varianza) para estabilizar distancias
pca = PCA(n_components=15, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ==============================================================================
# 2. Matriz de Enlace y Dendrograma Truncado (Ward)
# ==============================================================================
print("Calculando matriz de enlace Ward sobre 71 variables físicas...")
Z = linkage(X_pca, method="ward")

plt.figure(figsize=(13, 6))
dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90.0,
    leaf_font_size=10.0,
    show_contracted=True,
)
plt.axhline(
    y=Z[-3, 2],
    color="crimson",
    linestyle="--",
    linewidth=1.8,
    label="Corte k=4 Clústeres",
)
plt.title(
    "Dendrograma Jerárquico de Similitud Tímbrica (Ward)",
    fontweight="bold",
    fontsize=13,
)
plt.xlabel("Muestras Condensadas")
plt.ylabel("Distancia Euclidiana / Varianza Acumulada")
plt.legend(loc="upper right")
plt.tight_layout()

dendro_path = os.path.join(REPORTS_DIR, "hierarchical_dendrogram.png")
plt.savefig(dendro_path, dpi=300)
plt.show()
print(f"Dendrograma guardado en: {dendro_path}")

# ==============================================================================
# 3. Asignación de Clústeres (k=4) y Evaluación
# ==============================================================================
k_final = 4
agg = AgglomerativeClustering(n_clusters=k_final, linkage="ward")
labels_hier = agg.fit_predict(X_pca)

sil = silhouette_score(X_pca, labels_hier)
ch = calinski_harabasz_score(X_pca, labels_hier)
db = davies_bouldin_score(X_pca, labels_hier)
ari = adjusted_rand_score(true_labels, labels_hier)

print("\n" + "=" * 65)
print(f"       RESULTADOS CLUSTERING JERÁRQUICO WARD (k={k_final})")
print("=" * 65)
print(f"Silhouette Score (↑):         {sil:.4f}")
print(f"Calinski-Harabasz Index (↑):  {ch:.2f}")
print(f"Davies-Bouldin Index (↓):     {db:.4f}")
print(f"Adjusted Rand Index - ARI (↑): {ari:.4f}")
print("=" * 65)

# Matriz de Contingencia
df_eval = pd.DataFrame(
    {"Instrumento_Real": true_labels, "Cluster_Jerarquico": labels_hier}
)
contingency = pd.crosstab(
    df_eval["Instrumento_Real"],
    df_eval["Cluster_Jerarquico"],
    margins=True,
    margins_name="Total",
)
print("\n--- MATRIZ DE CONTINGENCIA (Validación Externa) ---")
print(contingency)
print("=" * 65)

# ==============================================================================
# 4. Exportar Resultados
# ==============================================================================
df_res = pd.DataFrame({
    "file_name": file_names,
    "true_instrument": true_labels,
    "cluster_hierarchical": labels_hier,
})
output_path = os.path.join(
    base_dir, "00_data", "processed", "results_hierarchical.csv"
)
df_res.to_csv(output_path, index=False)
print(f"Asignaciones guardadas en: {output_path}")