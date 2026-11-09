import warnings
import librosa
import numpy as np

warnings.filterwarnings("ignore")

NOTAS_CQT = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def extract_71_audio_features(y, sr=22050):
    """
    Extractor unificado de 71 características físico-acústicas:
    - Separación HPSS (1 ratio)
    - Croma CQT sobre componente armónica (12 medias + 12 desv. est.)
    - MFCCs (13 medias + 13 desv. est.)
    - Contraste Espectral en 6 bandas (7 medias + 7 desv. est.)
    - Descriptores de Forma: Centroide (2), Rolloff (2), Cruce por Cero (2)
    Total: 1 + 24 + 26 + 14 + 6 = 71 dimensiones.
    """
    # 1. Separación de Fuentes Armónica-Percusiva (HPSS)
    y_harm, y_perc = librosa.effects.hpss(y)
    energy_harm = float(np.sum(y_harm**2))
    energy_perc = float(np.sum(y_perc**2))
    hpss_ratio = (energy_perc - energy_harm) / (energy_perc + energy_harm + 1e-9)

    # 2. Croma Constant-Q (12 notas musicales)
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harm, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma_cqt, axis=1)
    chroma_std = np.std(chroma_cqt, axis=1)

    # 3. MFCC (13 coeficientes)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 4. Contraste Espectral (7 valles/crestas)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # 5. Momentos Espectrales y Dominio Temporal
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y)

    feature_vector = np.hstack([
        mfcc_mean,                      # 13
        mfcc_std,                       # 13
        chroma_mean,                    # 12
        chroma_std,                     # 12
        contrast_mean,                  # 7
        contrast_std,                   # 7
        np.mean(centroid), np.std(centroid),   # 2
        np.mean(rolloff), np.std(rolloff),     # 2
        np.mean(zcr), np.std(zcr),             # 2
        hpss_ratio                             # 1
    ])

    return feature_vector, chroma_mean, y_harm, y_perc