import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import hdbscan
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
# 1. Configuración de Rutas y Carga de Datos Limpios
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "00_data", "processed", "features_instruments_clean.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "05_reports")
PROCESSED_DIR = os.path.join(BASE_DIR, "00_data", "processed")

df = pd.read_csv(DATA_PATH)
feat_cols = [c for c in df.columns if c not in ["file_name", "true_instrument"]]
X_scaled = StandardScaler().fit_transform(df[feat_cols].values)

# Corrección Obs 6: Ajuste y evaluación estrictamente sobre 15 componentes ortogonales
pca_15d = PCA(n_components=15, random_state=42)
X_pca_15d = pca_15d.fit_transform(X_scaled)

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

true_labels = df["true_instrument"].values
file_names = df["file_name"].values

# ==============================================================================
# 2. Ajuste de Modelos Basados en Densidad (15D PCA)
# ==============================================================================
print("Ajustando DBSCAN y HDBSCAN sobre 15 componentes PCA...")

# Parámetros calibrados para 15 dimensiones acústicas
dbscan = DBSCAN(eps=3.4, min_samples=12)
labels_dbscan = dbscan.fit_predict(X_pca_15d)

clusterer_hdb = hdbscan.HDBSCAN(min_cluster_size=35, min_samples=10, gen_min_span_tree=True)
labels_hdbscan = clusterer_hdb.fit_predict(X_pca_15d)

modelos_densidad = {
    "DBSCAN (eps=3.4, min=12)": labels_dbscan,
    "HDBSCAN (min_size=35)": labels_hdbscan
}

# ==============================================================================
# 3. Evaluación de Métricas Excluyendo Ruido Acústico
# ==============================================================================
resultados = []

print("\n" + "=" * 80)
print("              COMPARATIVA DE MODELOS BASADOS EN DENSIDAD (15D PCA)")
print("=" * 80)

for nombre, labs in modelos_densidad.items():
    mask = labs != -1
    n_clusters = len(set(labs[mask]))
    n_ruido = int(np.sum(~mask))
    pct_ruido = round((n_ruido / len(labs)) * 100, 2)
    
    if n_clusters > 1:
        sil = round(float(silhouette_score(X_pca_15d[mask], labs[mask])), 4)
        ch = round(float(calinski_harabasz_score(X_pca_15d[mask], labs[mask])), 2)
        db = round(float(davies_bouldin_score(X_pca_15d[mask], labs[mask])), 4)
        ari = round(float(adjusted_rand_score(true_labels[mask], labs[mask])), 4)
        nmi = round(float(normalized_mutual_info_score(true_labels[mask], labs[mask])), 4)
    else:
        sil, ch, db, ari, nmi = 0.0, 0.0, 0.0, 0.0, 0.0

    resultados.append({
        "Modelo": nombre,
        "Clusters": n_clusters,
        "Ruido (Outliers)": f"{n_ruido} ({pct_ruido}%)",
        "Silueta (sin ruido)": sil,
        "Calinski-H": ch,
        "Davies-Bouldin": db,
        "ARI": ari,
        "NMI": nmi
    })

df_res_dens = pd.DataFrame(resultados)
print(df_res_dens.to_string(index=False))
print("=" * 80)

# ==============================================================================
# 4. Gráfico Comparativo 2D
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for ax, (nombre, labs) in zip([ax1, ax2], modelos_densidad.items()):
    mask_core = labs != -1
    ax.scatter(X_pca_2d[~mask_core, 0], X_pca_2d[~mask_core, 1], c='black', marker='x', alpha=0.3, label='Ruido (Outliers)')
    scatter = ax.scatter(X_pca_2d[mask_core, 0], X_pca_2d[mask_core, 1], c=labs[mask_core], cmap='viridis', s=25, alpha=0.7)
    ax.set_title(nombre, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle("Detección de Densidad y Filtrado de Ruido Acústico (15D PCA)", fontsize=13, fontweight="bold")
plt.tight_layout()
density_chart_path = os.path.join(REPORTS_DIR, "density_comparison.png")
plt.savefig(density_chart_path, dpi=300)
plt.close()

# Exportar resultados a CSV
df_out = pd.DataFrame({
    "file_name": file_names,
    "true_instrument": true_labels,
    "cluster_dbscan": labels_dbscan,
    "cluster_hdbscan": labels_hdbscan
})
output_csv = os.path.join(PROCESSED_DIR, "results_density.csv")
df_out.to_csv(output_csv, index=False)
print(f"Gráfico guardado en: {density_chart_path}")
print(f"Asignaciones exportadas a: {output_csv}")