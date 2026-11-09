import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

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

DATA_PATH = os.path.join(base_dir, "00_data", "processed", "features_instruments.csv")
REPORTS_DIR = os.path.join(base_dir, "05_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ["file_name", "true_instrument"]]
X_raw = df[feature_cols].values
y_labels = df["true_instrument"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"Dataset cargado: {X_scaled.shape[0]} muestras x {X_scaled.shape[1]} características.")

# ==============================================================================
# 2. Ajuste de Modelos de Reducción de Dimensionalidad a 2D
# ==============================================================================
print("\nEjecutando reducciones de dimensionalidad...")

# A) PCA Clásico
print("  1. PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# B) Truncated SVD (Álgebra Lineal pura)
print("  2. SVD Truncado...")
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_scaled)

# C) Factor Analysis (PPCA - Marco Probabilístico)
print("  3. Factor Analysis (PPCA)...")
fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)

# D) t-SNE (No Lineal, Local)
print("  4. t-SNE...")
tsne = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# E) UMAP (Variedad Topológica)
print("  5. UMAP...")
reducer_umap = umap.UMAP(n_neighbors=25, min_dist=0.3, n_components=2, random_state=42)
X_umap = reducer_umap.fit_transform(X_scaled)

# ==============================================================================
# 3. Lienzo Comparativo de 5 Paneles
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes_flat = axes.flatten()

modelos = [
    ("PCA (Lineal / Ortogonal)", X_pca, axes_flat[0]),
    ("SVD Truncado (Factorización)", X_svd, axes_flat[1]),
    ("Factor Analysis / PPCA (Probabilístico)", X_fa, axes_flat[2]),
    ("t-SNE (No Lineal / Estructura Local)", X_tsne, axes_flat[3]),
    ("UMAP (Topología Riemanniana)", X_umap, axes_flat[4]),
]

clases_unicas = sorted(list(set(y_labels)))
cmap = plt.colormaps["tab10"].resampled(len(clases_unicas))

for titulo, X_proj, ax in modelos:
    for idx, cls in enumerate(clases_unicas):
        mask = y_labels == cls
        ax.scatter(
            X_proj[mask, 0],
            X_proj[mask, 1],
            color=cmap(idx),
            label=cls,
            alpha=0.55,
            s=25
        )
    ax.set_title(titulo, fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

# El 6to subplot se usa para la leyenda explicativa consolidada
axes_flat[5].axis("off")
handles, labels = axes_flat[0].get_legend_handles_labels()
axes_flat[5].legend(
    handles,
    labels,
    loc="center",
    title="Familias Instrumentales",
    title_fontsize=12,
    fontsize=11,
    frameon=True,
    facecolor="#f8fafc"
)
axes_flat[5].text(
    0.1, 0.2,
    "Guía Metodológica:\n"
    "• PCA/SVD: Varianza global espectral.\n"
    "• PPCA/FA: Modela ruido gaussiano latente.\n"
    "• t-SNE: Revela sub-islas de agrupamiento.\n"
    "• UMAP: Base óptima para K-Means en web.",
    fontsize=10,
    color="#334155"
)

plt.suptitle(
    "Estudio Comparativo de Espacios Latentes: Métodos Lineales vs No Lineales",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()

output_fig = os.path.join(REPORTS_DIR, "dim_reduction_comparison.png")
plt.savefig(output_fig, dpi=300)
plt.show()
print(f"\nPanel comparativo guardado en: {output_fig}")