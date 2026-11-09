import os
import warnings
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Configuración de Rutas
# ==============================================================================
current_dir = os.path.abspath(os.path.dirname(__file__))
base_dir = current_dir
while base_dir and not os.path.exists(os.path.join(base_dir, '00_data')):
  parent = os.path.dirname(base_dir)
  if parent == base_dir:
    break
  base_dir = parent

DATA_PATH = os.path.join(
    base_dir, '00_data', 'processed', 'features_instruments.csv'
)
CUSTOM_AUDIO_DIR = os.path.join(base_dir, '04_inference', 'custom_audio')
REPORTS_DIR = os.path.join(base_dir, '05_reports')

# ==============================================================================
# 2. Carga y Entrenamiento del Modelo Base + Mapeo de Etiquetas
# ==============================================================================
print('Cargando base de conocimiento acústica...')
df = pd.read_csv(DATA_PATH)
feature_cols = [
    c for c in df.columns if c not in ['file_name', 'true_instrument']
]

X_train = df[feature_cols].values
y_true = df['true_instrument'].values

# Escalador
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Ajuste GMM
k_clusters = 4
gmm = GaussianMixture(
    n_components=k_clusters, covariance_type='full', random_state=42
)
train_labels = gmm.fit_predict(X_train_scaled)

# Mapeo semántico automático por moda/mayoría
cluster_to_instrument = {}
for k in range(k_clusters):
  mask = train_labels == k
  if np.sum(mask) > 0:
    predominante = pd.Series(y_true[mask]).mode()[0]
    cluster_to_instrument[k] = predominante
  else:
    cluster_to_instrument[k] = f'Cluster {k}'

print('\n--- MAPEO SEMÁNTICO DE CLÚSTERES ---')
for k, inst in cluster_to_instrument.items():
  print(f'  Clúster {k} -> Representa principalmente: {inst}')


# ==============================================================================
# 3. Extractor Acústico para Audio en Vivo
# ==============================================================================
def extract_single_audio_features(file_path):
  y, sr = librosa.load(file_path, sr=22050, mono=True)
  y, _ = librosa.effects.trim(y, top_db=25)

  if len(y) < sr * 0.2:
    raise ValueError(
        'El audio es demasiado corto o solo contiene silencio (< 0.2s).'
    )

  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
  mfcc_mean = np.mean(mfcc, axis=1)
  mfcc_std = np.std(mfcc, axis=1)

  spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
  spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
  zcr = librosa.feature.zero_crossing_rate(y)

  return np.hstack([
      mfcc_mean,
      mfcc_std,
      np.mean(spec_cent),
      np.std(spec_cent),
      np.mean(spec_rolloff),
      np.std(spec_rolloff),
      np.mean(zcr),
      np.std(zcr),
  ])


# ==============================================================================
# 4. Inferencia del Archivo Personal
# ==============================================================================
# Buscar el primer archivo .wav dentro de custom_audio
archivos_wav = [f for f in os.listdir(CUSTOM_AUDIO_DIR) if f.endswith('.wav')]

if not archivos_wav:
  print(f'\n[AVISO] No se encontró ningún archivo .wav en {CUSTOM_AUDIO_DIR}')
  print('Coloca un archivo de audio (ej: mi_audio.wav) y vuelve a ejecutar.')
  exit()

audio_target = os.path.join(CUSTOM_AUDIO_DIR, archivos_wav[0])
print(f"\nProcesando archivo de prueba: '{archivos_wav[0]}'...")

# Extracción y escalado
feats = extract_single_audio_features(audio_target)
feats_scaled = scaler.transform(feats.reshape(1, -1))

# Predicción probabilística
pred_cluster = gmm.predict(feats_scaled)[0]
probs = gmm.predict_proba(feats_scaled)[0]
instrumento_detectado = cluster_to_instrument[pred_cluster]

print('\n' + '=' * 60)
print('             RESULTADO DE LA CLASIFICACIÓN EN VIVO')
print('=' * 60)
print(f'Instrumento Predicho:  --> {instrumento_detectado} <--')
print(f'Clúster GMM Asignado:  Clúster {pred_cluster}')
print(f'Nivel de Certidumbre:  {probs[pred_cluster] * 100:.2f}%')
print('-' * 60)
print('Distribución de Probabilidad Tímbrica:')
for k in range(k_clusters):
  print(
      f'  {cluster_to_instrument[k]:<20} (Clúster {k}):'
      f' {probs[k] * 100:6.2f}%'
  )
print('=' * 60)

# ==============================================================================
# 5. Visualización Espacial 2D con tu Muestra Marcada
# ==============================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
sample_pca = pca.transform(feats_scaled)

plt.figure(figsize=(10, 6.5))
cmap = plt.colormaps['tab10'].resampled(k_clusters)

# Graficar datos base
for k in range(k_clusters):
  mask = train_labels == k
  plt.scatter(
      X_pca[mask, 0],
      X_pca[mask, 1],
      color=cmap(k),
      alpha=0.35,
      s=30,
      label=f'{cluster_to_instrument[k]} (C{k})',
  )

# Resaltar la muestra en vivo
plt.scatter(
    sample_pca[0, 0],
    sample_pca[0, 1],
    color='red',
    marker='*',
    s=350,
    edgecolor='black',
    linewidth=1.8,
    label=f'Muestra en Vivo ({archivos_wav[0]})',
    zorder=10,
)

# Flecha indicadora
plt.annotate(
    f'Tu Audio:\n{instrumento_detectado}',
    xy=(sample_pca[0, 0], sample_pca[0, 1]),
    xytext=(sample_pca[0, 0] + 1.2, sample_pca[0, 1] + 1.2),
    arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8),
)

plt.title('Inferencia Acústica: Ubicación de tu Audio en el Espacio Latente PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

inference_plot_path = os.path.join(REPORTS_DIR, 'live_audio_inference_pca.png')
plt.savefig(inference_plot_path, dpi=300)
plt.show()
print(f'Gráfico de inferencia guardado en: {inference_plot_path}')