import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Configuración de Rutas y Datos
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
MODELS_BUNDLE = os.path.join(base_dir, "web_app", "models_bundle")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_BUNDLE, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ["file_name", "true_instrument"]]
X_raw = df[feature_cols].values
true_labels = df["true_instrument"].values
file_names = df["file_name"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Reducción a 15 componentes para estimar matrices de covarianza estables
pca = PCA(n_components=15, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ==============================================================================
# 2. Criterios de Selección de Modelo: BIC y AIC
# ==============================================================================
k_range = range(2, 9)
bic_scores = []
aic_scores = []

print("Evaluando criterios de información bayesiana y de Akaike (BIC / AIC)...")
for k in k_range:
    gmm_temp = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm_temp.fit(X_pca)
    bic_scores.append(gmm_temp.bic(X_pca))
    aic_scores.append(gmm_temp.aic(X_pca))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_range, bic_scores, marker="o", label="BIC (Bayesian Information)", color="#2563eb", linewidth=2)
ax.plot(k_range, aic_scores, marker="s", label="AIC (Akaike Information)", color="#dc2626", linewidth=2)
ax.set_title("Curvas de Selección de Modelo GMM (Menor es Mejor)", fontweight="bold")
ax.set_xlabel("Número de Componentes Gaussianas (k)")
ax.set_ylabel("Puntuación de Información")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()

bic_path = os.path.join(REPORTS_DIR, "gmm_bic_aic_curve.png")
plt.savefig(bic_path, dpi=300)
plt.show()
print(f"Curvas BIC/AIC guardadas en: {bic_path}")

# ==============================================================================
# 3. Ajuste GMM Definitivo (k=4) y Soft Clustering
# ==============================================================================
k_final = 4
gmm = GaussianMixture(n_components=k_final, covariance_type="full", random_state=42)
gmm.fit(X_pca)

labels_gmm = gmm.predict(X_pca)
probs = gmm.predict_proba(X_pca)
certidumbre_media = np.mean(np.max(probs, axis=1)) * 100

sil = silhouette_score(X_pca, labels_gmm)
ari = adjusted_rand_score(true_labels, labels_gmm)

print("\n" + "=" * 65)
print(f"      RESULTADOS MODELO MIXTO GAUSSIANO - GMM (k={k_final})")
print("=" * 65)
print(f"Silhouette Score (↑):         {sil:.4f}")
print(f"Adjusted Rand Index - ARI (↑): {ari:.4f}")
print(f"Certidumbre Media Asignada:    {certidumbre_media:.2f}%")
print("=" * 65)

# Matriz de Contingencia
df_eval = pd.DataFrame({"Instrumento_Real": true_labels, "Cluster_GMM": labels_gmm})
contingency = pd.crosstab(df_eval["Instrumento_Real"], df_eval["Cluster_GMM"], margins=True, margins_name="Total")
print("\n--- MATRIZ DE CONTINGENCIA GMM ---")
print(contingency)
print("=" * 65)

# ==============================================================================
# 4. Guardado y Serialización para la Aplicación Web (Bundle)
# ==============================================================================
pca_2d_viz = PCA(n_components=2, random_state=42)
pca_2d_viz.fit(X_scaled)

joblib.dump(scaler, os.path.join(MODELS_BUNDLE, "scaler.joblib"))
joblib.dump(pca, os.path.join(MODELS_BUNDLE, "pca_15d.joblib"))
joblib.dump(pca_2d_viz, os.path.join(MODELS_BUNDLE, "pca_2d.joblib"))
joblib.dump(gmm, os.path.join(MODELS_BUNDLE, "gmm_model.joblib"))

# Mapeo semántico de clústeres
crosstab_clean = pd.crosstab(df_eval["Cluster_GMM"], df_eval["Instrumento_Real"])
cluster_mapping = {}
for cl in range(k_final):
    if cl in crosstab_clean.index:
        cluster_mapping[cl] = crosstab_clean.loc[cl].idxmax()
    else:
        cluster_mapping[cl] = f"Cluster_{cl}"

joblib.dump(cluster_mapping, os.path.join(MODELS_BUNDLE, "cluster_mapping.joblib"))

print("\n--- ARTEFACTOS SERIALIZADOS PARA DESPLIEGUE WEB ---")
print(f"Guardados en: {MODELS_BUNDLE}")
print("  • scaler.joblib")
print("  • pca_15d.joblib")
print("  • pca_2d.joblib")
print("  • gmm_model.joblib")
print(f"  • cluster_mapping.joblib -> {cluster_mapping}")

# Exportar CSV de probabilidades
df_probs = pd.DataFrame(probs, columns=[f"prob_cluster_{i}" for i in range(k_final)])
df_probs["cluster_asignado"] = labels_gmm
df_probs["file_name"] = file_names
df_probs["true_instrument"] = true_labels
out_gmm = os.path.join(base_dir, "00_data", "processed", "results_gmm.csv")
df_probs.to_csv(out_gmm, index=False)
print(f"\nResultados y probabilidades exportadas a: {out_gmm}")