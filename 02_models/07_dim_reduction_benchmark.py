import os
import matplotlib
matplotlib.use('Agg')  # Guarda directo en 05_reports/ sin bloquear consola
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Configuración de Rutas y Datos Limpios
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "00_data", "processed", "features_instruments_clean.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "05_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feat_cols = [c for c in df.columns if c not in ["file_name", "true_instrument"]]
X_scaled = StandardScaler().fit_transform(df[feat_cols].values)
y_true = df["true_instrument"].values

color_map = {
    "Sound_Drum": "#ef4444",    # Rojo
    "Sound_Guitar": "#3b82f6",  # Azul
    "Sound_Piano": "#10b981"    # Verde
}

# ==============================================================================
# 2. Ejecución de Métodos Lineales y No Lineales
# ==============================================================================
print("Calculando proyecciones 2D para benchmark de reducción...")

metodos = {
    "1. PCA (Varianza Global)": PCA(n_components=2, random_state=42).fit_transform(X_scaled),
    "2. Truncated SVD (Espacio Latente)": TruncatedSVD(n_components=2, random_state=42).fit_transform(X_scaled),
    "3. Factor Analysis (Covarianza)": FactorAnalysis(n_components=2, random_state=42).fit_transform(X_scaled),
    # Corrección: max_iter compatible con versiones modernas de scikit-learn
    "4. t-SNE (Probabilidad Local)": TSNE(n_components=2, random_state=42, perplexity=35, max_iter=1000).fit_transform(X_scaled),
    "5. UMAP (Topología Riemanniana)": umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(X_scaled)
}

# ==============================================================================
# 3. Generación del Lienzo Comparativo Multimétodo
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (nombre, coords) in enumerate(metodos.items()):
    ax = axes[i]
    for inst, col in color_map.items():
        mask = y_true == inst
        ax.scatter(coords[mask, 0], coords[mask, 1], label=inst, c=col, alpha=0.55, s=20, edgecolors='none')
    ax.set_title(nombre, fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

# Cuadrante 6: Gráfica de Varianza Acumulada de PCA
pca_full = PCA(n_components=15, random_state=42).fit(X_scaled)
var_acum = pca_full.explained_variance_ratio_.cumsum() * 100

axes[5].plot(range(1, 16), var_acum, marker='o', color='#8b5cf6', linewidth=2)
axes[5].axhline(y=var_acum[14], color='red', linestyle='--', label=f'15 Comp: {var_acum[14]:.1f}% Varianza')
axes[5].set_title("Varianza Acumulada PCA (1 a 15 Comp.)", fontweight="bold", fontsize=11)
axes[5].set_xlabel("Número de Componentes")
axes[5].set_ylabel("Varianza Explicada (%)")
axes[5].grid(True, alpha=0.3)
axes[5].legend(loc="lower right")

plt.suptitle("Benchmark de Reducción Dimensional sobre 71 Variables Físicas (Dataset Saneado)", fontsize=14, fontweight="bold")
plt.tight_layout()

output_fig = os.path.join(REPORTS_DIR, "dim_reduction_comparison.png")
plt.savefig(output_fig, dpi=300)
plt.close()

print(f"Lienzo comparativo guardado exitosamente en: {output_fig}")