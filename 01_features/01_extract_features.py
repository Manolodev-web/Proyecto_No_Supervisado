import os
import warnings
import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. Configuración de Rutas
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "00_data", "raw", "Metadata_Train.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "00_data", "raw", "Train_submission")
OUTPUT_DIR = os.path.join(BASE_DIR, "00_data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "features_instruments.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de metadatos: {METADATA_PATH}")
if not os.path.exists(AUDIO_DIR):
    raise FileNotFoundError(f"No se encontró la carpeta de audios: {AUDIO_DIR}")

print("Cargando metadatos...")
df_meta = pd.read_csv(METADATA_PATH)

col_file = [c for c in df_meta.columns if any(k in c.lower() for k in ["file", "audio", "name", "id"])][0]
col_label = [c for c in df_meta.columns if any(k in c.lower() for k in ["class", "label", "instrument", "target"])][0]

print(f"Columnas detectadas -> Archivo: [{col_file}] | Etiqueta: [{col_label}]")
print(f"Total de registros a procesar: {len(df_meta)}")

# ==============================================================================
# 2. Extractor Acústico: CQT, HPSS, MFCCs y Dinámica Espectral
# ==============================================================================
def extract_audio_physics(file_path):
    # Carga a 22050 Hz mono
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    
    if len(y) < sr * 0.25:
        return None

    # A) Separación Armónica-Percusiva (HPSS)
    y_harm, y_perc = librosa.effects.hpss(y)
    energy_harm = float(np.sum(y_harm ** 2))
    energy_perc = float(np.sum(y_perc ** 2))
    hpss_ratio = (energy_perc - energy_harm) / (energy_perc + energy_harm + 1e-9)

    # B) Cromagrama CQT (Transformada Constant-Q: 12 semitonos musicales)
    # Mayor resolución en graves que la FFT estándar
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harm, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma_cqt, axis=1)
    chroma_std = np.std(chroma_cqt, axis=1)

    # C) Envolvente Tímbrica (MFCCs: 13 coeficientes)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # D) Contraste Espectral (6 bandas de frecuencia: picos vs valles)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # E) Descriptores Espectrales Clásicos
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y)

    return np.hstack([
        mfcc_mean,                     # 13
        mfcc_std,                      # 13
        chroma_mean,                   # 12 (Notas C, C#, D... B con CQT)
        chroma_std,                    # 12
        contrast_mean,                 # 7
        contrast_std,                  # 7
        np.mean(centroid), np.std(centroid),    # 2
        np.mean(rolloff), np.std(rolloff),      # 2
        np.mean(zcr), np.std(zcr),              # 2
        hpss_ratio                              # 1
    ])

# ==============================================================================
# 3. Nombres de Columnas Tabulares
# ==============================================================================
notas = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
feature_names = (
    [f"mfcc_mean_{i}" for i in range(13)] +
    [f"mfcc_std_{i}" for i in range(13)] +
    [f"chroma_cqt_mean_{n}" for n in notas] +
    [f"chroma_cqt_std_{n}" for n in notas] +
    [f"contrast_mean_{b}" for b in range(7)] +
    [f"contrast_std_{b}" for b in range(7)] +
    ["spec_cent_mean", "spec_cent_std", "spec_rolloff_mean", "spec_rolloff_std", "zcr_mean", "zcr_std", "hpss_ratio"]
)

total_features = len(feature_names)
print(f"Vector acústico configurado: {total_features} dimensiones.")

# ==============================================================================
# 4. Procesamiento del Lote de Audios
# ==============================================================================
records = []
faltantes = 0
procesados = 0

print("\nIniciando extracción por lotes...")
for idx, row in df_meta.iterrows():
    fname = str(row[col_file]).strip()
    if not fname.lower().endswith(".wav"):
        fname += ".wav"

    fpath = os.path.join(AUDIO_DIR, fname)
    if not os.path.exists(fpath):
        faltantes += 1
        continue

    try:
        feats = extract_audio_physics(fpath)
        if feats is not None and len(feats) == total_features:
            record = {
                "file_name": fname,
                "true_instrument": str(row[col_label]).strip()
            }
            for col, val in zip(feature_names, feats):
                record[col] = val
            records.append(record)
            procesados += 1

            if procesados % 100 == 0:
                print(f"  -> {procesados} audios extraídos con éxito...")
    except Exception:
        continue

# ==============================================================================
# 5. Exportar CSV Final
# ==============================================================================
df_final = pd.DataFrame(records)
df_final.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 65)
print("EXTRACCIÓN FINALIZADA CON ÉXITO")
print("=" * 65)
print(f"Total registros guardados: {len(df_final)}")
print(f"Archivos omitidos/no encontrados: {faltantes}")
print(f"Dimensiones de la matriz: {df_final.shape}")
print("\nDistribución de instrumentos:")
print(df_final["true_instrument"].value_counts())
print(f"\nArchivo guardado en: {OUTPUT_CSV}")
print("=" * 65)