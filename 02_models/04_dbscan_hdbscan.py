import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import hdbscan

# ==============================================================================
# 1. Carga y Normalización del Espacio Acústico
# ==============================================================================
current_dir = os.path.abspath(os.path.dirname(__file__))
base_dir = current_dir
while base_dir and not os.path.exists(os.path.join(base_dir, "00_data")):
    parent = os.path.dirname(base_dir)
    if parent == base_dir:
        break
    base_dir = parent

DATA_PATH = os.path.join(base_dir, "00_data", "processed", "features_instruments.csv")
REPORTS_DIR = os.path.join(base_dir, "05_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ["file_name", "true_instrument"]]
X_raw = df[feature_cols].values
true_labels = df["true_instrument"].values
file_names = df["file_name"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Reducción a 10 componentes PCA para preservar densidad
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ==============================================================================
# 2. Ajuste DBSCAN vs HDBSCAN
# ==============================================================================
print("Ajustando modelos basados en densidad...")

# A) DBSCAN (Densidad global fija)
dbscan_model = DBSCAN(eps=2.8, min_samples=15)
labels_db = dbscan_model.fit_predict(X_pca)

# B) HDBSCAN (Densidad jerárquica variable)
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10, gen_min_span_tree=True)
labels_hdb = hdbscan_model.fit_predict(X_pca)

# ==============================================================================
# 3. Métricas y Validación
# ==============================================================================
def evaluar_densidad(nombre, labels, X_data):
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_ruido = int(np.sum(labels == -1))
    pct_ruido = round((n_ruido / len(labels)) * 100, 2)
    
    if n_clusters > 1 and np.sum(mask) > n_clusters:
        sil = round(float(silhouette_score(X_data[mask], labels[mask])), 4)
        ch = round(float(calinski_harabasz_score(X_data[mask], labels[mask])), 2)
    else:
        sil, ch = 0.0, 0.0
        
    return {
        "Modelo": nombre,
        "Clusters": n_clusters,
        "Puntos Ruido": f"{n_ruido} ({pct_ruido}%)",
        "Silueta (sin ruido)": sil,
        "Calinski-H": ch
    }

tabla_metricas = [
    evaluar_densidad("DBSCAN (eps=2.8)", labels_db, X_pca),
    evaluar_densidad("HDBSCAN (min_size=40)", labels_hdb, X_pca)
]

print("\n" + "=" * 70)
print("            COMPARATIVA DE MODELOS BASADOS EN DENSIDAD")
print("=" * 70)
print(pd.DataFrame(tabla_metricas).to_string(index=False))
print("=" * 70)

# ==============================================================================
# 4. Visualización Comparativa 2D
# ==============================================================================
pca_2d = PCA(n_components=2, random_state=42)
X_viz = pca_2d.fit_transform(X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

def plot_density_scatter(ax, labels, titulo):
    unique_labs = sorted(list(set(labels)))
    n_validos = len([l for l in unique_labs if l != -1])
    cmap = plt.colormaps['tab10'].resampled(max(n_validos, 1))
    
    c_idx = 0
    for lab in unique_labs:
        mask = labels == lab
        if lab == -1:
            ax.scatter(X_viz[mask, 0], X_viz[mask, 1], c='black', marker='x', s=35, alpha=0.5, label=f'Ruido ({np.sum(mask)})')
        else:
            ax.scatter(X_viz[mask, 0], X_viz[mask, 1], color=cmap(c_idx), s=28, alpha=0.6, label=f'Clúster {lab}')
            c_idx += 1
            
    ax.set_title(titulo, fontweight='bold', fontsize=11)
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower left', fontsize=8)

plot_density_scatter(ax1, labels_db, f"DBSCAN: {tabla_metricas[0]['Clusters']} Clusters | {tabla_metricas[0]['Puntos Ruido']}")
plot_density_scatter(ax2, labels_hdb, f"HDBSCAN: {tabla_metricas[1]['Clusters']} Clusters | {tabla_metricas[1]['Puntos Ruido']}")

plt.suptitle("Clustering Basado en Densidad y Detección de Ruido Acústico", fontsize=13, fontweight='bold')
plt.tight_layout()

density_img = os.path.join(REPORTS_DIR, "density_comparison.png")
plt.savefig(density_img, dpi=300)
plt.show()
print(f"Gráfico de densidad guardado en: {density_img}")

# ==============================================================================
# 5. Exportar Resultados
# ==============================================================================
df_out = pd.DataFrame({
    "file_name": file_names,
    "true_instrument": true_labels,
    "cluster_dbscan": labels_db,
    "cluster_hdbscan": labels_hdb
})
out_path = os.path.join(base_dir, "00_data", "processed", "results_density.csv")
df_out.to_csv(out_path, index=False)
print(f"Asignaciones exportadas a: {out_path}")