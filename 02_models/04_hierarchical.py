import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
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
DATA_PATH = os.path.join(BASE_DIR, "00_data", "processed", "features_instruments_clean.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "05_reports")
PROCESSED_DIR = os.path.join(BASE_DIR, "00_data", "processed")

df = pd.read_csv(DATA_PATH)
feat_cols = [c for c in df.columns if c not in ["file_name", "true_instrument"]]
X_scaled = StandardScaler().fit_transform(df[feat_cols].values)

# PCA unificado a 15 componentes
pca_15d = PCA(n_components=15, random_state=42)
X_pca = pca_15d.fit_transform(X_scaled)

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

true_labels = df["true_instrument"].values
file_names = df["file_name"].values

# ==============================================================================
# 2. Matriz de Enlace Ward y Dendrograma
# ==============================================================================
print("Calculando matriz de enlace Ward sobre 15 componentes PCA...")
Z = linkage(X_pca, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=9., show_contracted=True)
plt.title("Dendrograma de Clustering Jerárquico (Enlace Ward)", fontweight="bold")
plt.xlabel("Índice de Muestra o Tamaño de Clúster")
plt.ylabel("Distancia de Fusión (Varianza)")
plt.axhline(y=115, color='crimson', linestyle='--', label='Corte k=3')
plt.legend()
plt.tight_layout()

dendro_path = os.path.join(REPORTS_DIR, "hierarchical_dendrogram.png")
plt.savefig(dendro_path, dpi=300)
plt.close()
print(f"Dendrograma guardado en: {dendro_path}")

# ==============================================================================
# 3. Asignación de Clústeres (k=3) y Métricas
# ==============================================================================
k_final = 3
hier_model = AgglomerativeClustering(n_clusters=k_final, linkage='ward')
cluster_labels = hier_model.fit_predict(X_pca)

sil = silhouette_score(X_pca, cluster_labels)
ch = calinski_harabasz_score(X_pca, cluster_labels)
db = davies_bouldin_score(X_pca, cluster_labels)
ari = adjusted_rand_score(true_labels, cluster_labels)
nmi = normalized_mutual_info_score(true_labels, cluster_labels)

print("\n" + "=" * 65)
print(f"    RESULTADOS CLUSTERING JERÁRQUICO WARD (k={k_final})")
print("=" * 65)
print(f"Silhouette Score (↑):              {sil:.4f}")
print(f"Calinski-Harabasz Index (↑):       {ch:.2f}")
print(f"Davies-Bouldin Index (↓):          {db:.4f}")
print(f"Adjusted Rand Index - ARI (↑):      {ari:.4f}")
print(f"Normalized Mutual Info - NMI (↑):  {nmi:.4f}")
print("=" * 65)

# Matriz de Contingencia
df_eval = pd.DataFrame({"Instrumento_Real": true_labels, "Cluster_Jerarquico": cluster_labels})
contingency = pd.crosstab(df_eval["Instrumento_Real"], df_eval["Cluster_Jerarquico"], margins=True, margins_name="Total")
print("\n--- MATRIZ DE CONTINGENCIA JERÁRQUICA ---")
print(contingency)
print("=" * 65)

# Gráfico de Clústeres en PCA 2D
plt.figure(figsize=(7, 5))
for cl in range(k_final):
    mask = cluster_labels == cl
    plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], label=f"Clúster {cl}", alpha=0.6, s=25)
plt.title(f"Clustering Jerárquico (k={k_final}) proyectado en PCA 2D", fontweight="bold")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
cluster_plot_path = os.path.join(REPORTS_DIR, "hierarchical_pca_clusters.png")
plt.savefig(cluster_plot_path, dpi=300)
plt.close()

# Exportación de Resultados
df_res = pd.DataFrame({
    "file_name": file_names,
    "true_instrument": true_labels,
    "cluster_hierarchical": cluster_labels
})
df_res.to_csv(os.path.join(PROCESSED_DIR, "results_hierarchical.csv"), index=False)
print(f"Asignaciones guardadas en: 00_data/processed/results_hierarchical.csv")