# Acoustic Unsupervised MIR Platform: Instrument Clustering & Source Separation

Este repositorio contiene la implementación completa del proyecto de **Aprendizaje No Supervisado** para la clasificación tímbrica y separación de fuentes de señales musicales (*Piano, Guitarra y Percusión*).

## Estructura del Pipeline

El proyecto está diseñado para ejecutarse de forma secuencial y determinista:

1. `01_features/audio_features.py`: Módulo canónico de extracción física de 71 dimensiones (HPSS + CQT + MFCC + Espectral).
2. `01_features/02_clean_dataset.py`: Auditoría forense de duplicación y generación de `features_instruments_clean.csv` (1,921 muestras).
3. `02_models/03_kmeans_clustering.py`: K-Means calibrado en $k=3$ (codo, silueta y serialización a `models_bundle`).
4. `02_models/04_hierarchical.py`: Clustering Jerárquico Aglomerativo con enlace Ward y dendrograma.
5. `02_models/05_density_dbscan_hdbscan.py`: Modelos de densidad (DBSCAN y HDBSCAN) sobre 15 dimensiones PCA.
6. `02_models/06_gmm_probabilistic.py`: Modelos Mixtos Gaussianos (BIC/AIC, soft clustering y exportación web).
7. `02_models/07_dim_reduction_benchmark.py`: Benchmark 2D comparando PCA, SVD, Factor Analysis, t-SNE y UMAP.
8. `03_evaluation/08_benchmark_metrics.py`: Consolidación oficial de métricas (Silueta, Calinski-H, Davies-Bouldin, ARI y NMI).
9. `04_inference/09_predict_instrument.py`: Inferencia rápida por consola para cualquier archivo `.mp3` o `.wav`.
10. `web_app/app.py`: Plataforma interactiva en Flask con análisis temporal y separación de fuentes.

## Instalación y Reproducción

```bash
# 1. Clonar el repositorio
git clone [https://github.com/Manolodev-web/Proyecto_No_Supervisado.git](https://github.com/Manolodev-web/Proyecto_No_Supervisado.git)
cd Proyecto_No_Supervisado

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Ejecutar pipeline de modelado y evaluación
python 01_features/02_clean_dataset.py
python 02_models/03_kmeans_clustering.py
python 02_models/04_hierarchical.py
python 02_models/05_density_dbscan_hdbscan.py
python 02_models/06_gmm_probabilistic.py
python 02_models/07_dim_reduction_benchmark.py
python 03_evaluation/08_benchmark_metrics.py

# 4. Probar inferencia por consola
python 04_inference/09_predict_instrument.py --audio 04_inference/custom_audio/mi_prueba.mp3

# 5. Iniciar la aplicación web
cd web_app
python app.py