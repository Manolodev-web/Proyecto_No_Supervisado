import os
import joblib
import matplotlib
matplotlib.use('Agg')  # Gráficos en segundo plano sin bloquear terminal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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
# 1. Configuración de Rutas y Datos Limpios
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "00_data", "processed", "features_instruments_clean.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "05_reports")
MODELS_BUNDLE = os.path.join(BASE_DIR, "web_app", "models_bundle")
PROCESSED_DIR = os.path.join(BASE_DIR, "00_data", "processed")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_BUNDLE, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feat_cols = [c for c in df.columns if c not in ["file_name", "true_instrument"]]
X_raw = df[feat_cols].values
true_labels = df["true_instrument"].values
file_names = df["file_name"].values

# Estandarización y PCA a 15 componentes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca_15d = PCA(n_components=15, random_state=42)
X_pca = pca_15d.fit_transform(X_scaled)

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# ==============================================================================
# 2. Análisis del Rango de k (Método del Codo y Silueta)
# ==============================================================================
k_range = range(2, 9)
inercias = []
siluetas = []

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=15, random_state=42)
    labels = km.fit_predict(X_pca)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_pca, labels))

# Gráfico diagnóstico
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(k_range, inercias, marker='o', color='#2563eb', linewidth=2)
ax1.set_title("Método del Codo (Inercia Acústica)", fontweight="bold")
ax1.set_xlabel("Número de Clústeres (k)")
ax1.set_ylabel("Inercia (SSE)")
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, siluetas, marker='s', color='#10b981', linewidth=2)
ax2.axvline(x=3, color='red', linestyle='--', label='k=3 (Óptimo Físico)')
ax2.set_title("Coeficiente de Silueta por k", fontweight="bold")
ax2.set_xlabel("Número de Clústeres (k)")
ax2.set_ylabel("Silueta Media")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
elbow_path = os.path.join(REPORTS_DIR, "kmeans_elbow_silhouette.png")
plt.savefig(elbow_path, dpi=300)
plt.close()

# ==============================================================================
# 3. Ajuste Final K-Means (k=3)
# ==============================================================================
k_final = 3
kmeans_final = KMeans(n_clusters=k_final, init="k-means++", n_init=25, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_pca)

sil = silhouette_score(X_pca, cluster_labels)
ch = calinski_harabasz_score(X_pca, cluster_labels)
db = davies_bouldin_score(X_pca, cluster_labels)
ari = adjusted_rand_score(true_labels, cluster_labels)
nmi = normalized_mutual_info_score(true_labels, cluster_labels)

print("\n" + "=" * 65)
print(f"      K-MEANS CLUSTERING (k={k_final}) - DATASET DEDUPLICADO")
print("=" * 65)
print(f"Silhouette Score (↑):              {sil:.4f}")
print(f"Calinski-Harabasz Index (↑):       {ch:.2f}")
print(f"Davies-Bouldin Index (↓):          {db:.4f}")
print(f"Adjusted Rand Index - ARI (↑):      {ari:.4f}")
print(f"Normalized Mutual Info - NMI (↑):  {nmi:.4f}")
print("=" * 65)

# Matriz de Contingencia
df_eval = pd.DataFrame({"Instrumento_Real": true_labels, "Cluster_KMeans": cluster_labels})
contingency = pd.crosstab(df_eval["Instrumento_Real"], df_eval["Cluster_KMeans"], margins=True, margins_name="Total")
print("\n--- MATRIZ DE CONTINGENCIA (Validación Externa) ---")
print(contingency)
print("=" * 65)

# Interpretación Post-Hoc: Asignación semántica clúster -> instrumento
crosstab_clean = pd.crosstab(df_eval["Cluster_KMeans"], df_eval["Instrumento_Real"])
kmeans_mapping = {cl: crosstab_clean.loc[cl].idxmax() for cl in range(k_final)}
print("\n[NOTA METODOLÓGICA] Mapeo Semántico Post-Hoc (interpretación posterior):")
for cl, inst in kmeans_mapping.items():
    print(f"   • Clúster {cl} -> {inst}")

# Gráfico 2D de clústeres
plt.figure(figsize=(7, 5))
for cl in range(k_final):
    mask = cluster_labels == cl
    plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], label=f"C{cl} ({kmeans_mapping[cl]})", alpha=0.6, s=25)
plt.title(f"K-Means (k={k_final}) proyectado en PCA 2D", fontweight="bold")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
pca_cluster_path = os.path.join(REPORTS_DIR, "kmeans_pca_clusters.png")
plt.savefig(pca_cluster_path, dpi=300)
plt.close()

# ==============================================================================
# 4. Serialización de Artefactos y Resultados
# ==============================================================================
joblib.dump(scaler, os.path.join(MODELS_BUNDLE, "scaler.joblib"))
joblib.dump(pca_15d, os.path.join(MODELS_BUNDLE, "pca_15d.joblib"))
joblib.dump(pca_2d, os.path.join(MODELS_BUNDLE, "pca_2d.joblib"))
joblib.dump(kmeans_final, os.path.join(MODELS_BUNDLE, "kmeans_model.joblib"))
joblib.dump(kmeans_mapping, os.path.join(MODELS_BUNDLE, "kmeans_mapping.joblib"))

df_res = pd.DataFrame({
    "file_name": file_names,
    "true_instrument": true_labels,
    "cluster_kmeans": cluster_labels
})
df_res.to_csv(os.path.join(PROCESSED_DIR, "results_kmeans.csv"), index=False)
print(f"\nResultados guardados en: 00_data/processed/results_kmeans.csv")
print(f"Artefactos exportados a: web_app/models_bundle/")