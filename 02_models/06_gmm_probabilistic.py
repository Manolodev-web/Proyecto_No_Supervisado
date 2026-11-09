import os
import joblib
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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Configuración de Rutas y Datos Limpios
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "00_data", "processed", "features_instruments_clean.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "05_reports")
MODELS_BUNDLE = os.path.join(BASE_DIR, "web_app", "models_bundle")
PROCESSED_DIR = os.path.join(BASE_DIR, "00_data", "processed")

df = pd.read_csv(DATA_PATH)
feat_cols = [c for c in df.columns if c not in ["file_name", "true_instrument"]]
X_raw = df[feat_cols].values
true_labels = df["true_instrument"].values
file_names = df["file_name"].values

# Transformación estándar y reducción unificada a 15D PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca_15d = PCA(n_components=15, random_state=42)
X_pca_15d = pca_15d.fit_transform(X_scaled)

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# ==============================================================================
# 2. Selección de Componentes por Criterios de Información (BIC / AIC)
# ==============================================================================
print("Evaluando criterios de información bayesiana y de Akaike (BIC / AIC)...")
k_range = range(2, 8)
bic_scores = []
aic_scores = []

for k in k_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=10)
    gmm.fit(X_pca_15d)
    bic_scores.append(gmm.bic(X_pca_15d))
    aic_scores.append(gmm.aic(X_pca_15d))

plt.figure(figsize=(8, 4.5))
plt.plot(k_range, bic_scores, marker='o', color='#2563eb', label='BIC (Bayesian Information)')
plt.plot(k_range, aic_scores, marker='s', color='#dc2626', label='AIC (Akaike Information)')
plt.axvline(x=3, color='green', linestyle='--', label='k=3 (Familias Reales)')
plt.title("Curvas de Selección de Modelo GMM (Menor es Mejor)", fontweight="bold")
plt.xlabel("Número de Componentes Gaussianas (k)")
plt.ylabel("Puntuación de Información")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

bic_path = os.path.join(REPORTS_DIR, "gmm_bic_aic_curve.png")
plt.savefig(bic_path, dpi=300)
plt.close()
print(f"Curvas BIC/AIC guardadas en: {bic_path}")

# ==============================================================================
# 3. Ajuste GMM Final (k=3) y Evaluación
# ==============================================================================
k_final = 3
gmm_final = GaussianMixture(n_components=k_final, covariance_type='full', random_state=42, n_init=20)
gmm_final.fit(X_pca_15d)

cluster_labels = gmm_final.predict(X_pca_15d)
probs = gmm_final.predict_proba(X_pca_15d)
confianza_media = np.mean(np.max(probs, axis=1)) * 100

sil = silhouette_score(X_pca_15d, cluster_labels)
ch = calinski_harabasz_score(X_pca_15d, cluster_labels)
db = davies_bouldin_score(X_pca_15d, cluster_labels)
ari = adjusted_rand_score(true_labels, cluster_labels)
nmi = normalized_mutual_info_score(true_labels, cluster_labels)

print("\n" + "=" * 65)
print(f"       RESULTADOS MODELO MIXTO GAUSSIANO - GMM (k={k_final})")
print("=" * 65)
print(f"Silhouette Score (↑):              {sil:.4f}")
print(f"Calinski-Harabasz Index (↑):       {ch:.2f}")
print(f"Davies-Bouldin Index (↓):          {db:.4f}")
print(f"Adjusted Rand Index - ARI (↑):      {ari:.4f}")
print(f"Normalized Mutual Info - NMI (↑):  {nmi:.4f}")
print(f"Certidumbre Media Asignada:        {confianza_media:.2f}%")
print("=" * 65)

# Matriz de Contingencia y Mapeo Post-Hoc
df_eval = pd.DataFrame({"Instrumento_Real": true_labels, "Cluster_GMM": cluster_labels})
contingency = pd.crosstab(df_eval["Instrumento_Real"], df_eval["Cluster_GMM"], margins=True, margins_name="Total")
print("\n--- MATRIZ DE CONTINGENCIA GMM ---")
print(contingency)
print("=" * 65)

crosstab_clean = pd.crosstab(df_eval["Cluster_GMM"], df_eval["Instrumento_Real"])
cluster_mapping = {cl: crosstab_clean.loc[cl].idxmax() for cl in range(k_final)}

print("\n[NOTA METODOLÓGICA] Mapeo Semántico Post-Hoc (interpretación posterior):")
for cl, inst in cluster_mapping.items():
    print(f"   • Clúster {cl} -> {inst}")

# ==============================================================================
# 4. Serialización de Artefactos para Consola y Web
# ==============================================================================
joblib.dump(scaler, os.path.join(MODELS_BUNDLE, "scaler.joblib"))
joblib.dump(pca_15d, os.path.join(MODELS_BUNDLE, "pca_15d.joblib"))
joblib.dump(pca_2d, os.path.join(MODELS_BUNDLE, "pca_2d.joblib"))
joblib.dump(gmm_final, os.path.join(MODELS_BUNDLE, "gmm_model.joblib"))
joblib.dump(cluster_mapping, os.path.join(MODELS_BUNDLE, "cluster_mapping.joblib"))

print("\n--- ARTEFACTOS SERIALIZADOS PARA DESPLIEGUE ---")
print(f"Guardados en: {MODELS_BUNDLE}")
for art in ["scaler.joblib", "pca_15d.joblib", "pca_2d.joblib", "gmm_model.joblib", "cluster_mapping.joblib"]:
    print(f"  • {art}")

# Exportar asignaciones y probabilidades
df_res = pd.DataFrame({
    "file_name": file_names,
    "true_instrument": true_labels,
    "cluster_asignado": cluster_labels,
    "prob_maxima": np.max(probs, axis=1)
})
res_path = os.path.join(PROCESSED_DIR, "results_gmm.csv")
df_res.to_csv(res_path, index=False)
print(f"\nResultados exportados a: {res_path}")