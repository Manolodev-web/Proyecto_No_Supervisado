import os
import sys
import argparse
import joblib
import librosa
import numpy as np

# Vincular módulo común de extracción
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "01_features"))
from audio_features import extract_71_audio_features, NOTAS_CQT

MODELS_BUNDLE = os.path.join(BASE_DIR, "web_app", "models_bundle")

def predict_audio(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: El archivo '{audio_path}' no existe.")
        return

    print("=" * 65)
    print("      INFERENCIA ACÚSTICA POR CONSOLA (MODELO 15D PCA)")
    print("=" * 65)
    print(f"Archivo de audio: {os.path.basename(audio_path)}")

    # 1. Cargar modelos serializados unificados
    scaler = joblib.load(os.path.join(MODELS_BUNDLE, "scaler.joblib"))
    pca_15d = joblib.load(os.path.join(MODELS_BUNDLE, "pca_15d.joblib"))
    kmeans_model = joblib.load(os.path.join(MODELS_BUNDLE, "kmeans_model.joblib"))
    kmeans_mapping = joblib.load(os.path.join(MODELS_BUNDLE, "kmeans_mapping.joblib"))
    gmm_model = joblib.load(os.path.join(MODELS_BUNDLE, "gmm_model.joblib"))
    cluster_mapping = joblib.load(os.path.join(MODELS_BUNDLE, "cluster_mapping.joblib"))

    # 2. Cargar audio y extraer 71 características deterministas
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)

    feats, chroma_vector, _, _ = extract_71_audio_features(y, sr)

    # 3. Transformación estándar y proyección latente a 15D
    x_scaled = scaler.transform(feats.reshape(1, -1))
    x_15d = pca_15d.transform(x_scaled)

    # 4. Inferencia K-Means (Partición dura)
    c_km = int(kmeans_model.predict(x_15d)[0])
    inst_km = kmeans_mapping.get(c_km, f"Clúster {c_km}")

    # 5. Inferencia GMM (Probabilística / Blanda)
    probs_gmm = gmm_model.predict_proba(x_15d)[0]
    c_gmm = int(np.argmax(probs_gmm))
    inst_gmm = cluster_mapping.get(c_gmm, f"Clúster {c_gmm}")
    confianza = float(np.max(probs_gmm) * 100)

    # 6. Detección tonal CQT
    top_indices = np.argsort(chroma_vector)[::-1][:3]
    top_notas = [f"{NOTAS_CQT[i]} ({chroma_vector[i]:.2f})" for i in top_indices]

    print("\n--- RESULTADOS DE INFERENCIA NO SUPERVISADA ---")
    print(f"1. Predicción K-Means:         {inst_km} (Clúster {c_km})")
    print(f"2. Predicción GMM:             {inst_gmm} (Clúster {c_gmm})")
    print(f"   Certidumbre GMM:            {confianza:.2f}%")
    print(f"3. Notas Dominantes (CQT):     {' | '.join(top_notas)}")
    print("\nDistribución de Probabilidades GMM:")
    for i, p in enumerate(probs_gmm):
        nombre = cluster_mapping.get(i, f"Clúster {i}")
        print(f"   • {nombre} (C{i}): {p * 100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificación acústica no supervisada de un audio.")
    parser.add_argument("--audio", type=str, default=os.path.join(BASE_DIR, "04_inference", "custom_audio", "mi_prueba.mp3"),
                        help="Ruta al archivo .wav o .mp3")
    args = parser.parse_args()
    predict_audio(args.audio)