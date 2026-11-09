import os
import joblib
import librosa
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# ==============================================================================
# Configuración de Rutas y Carga de Modelos Serializados
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(BASE_DIR, "models_bundle")
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Cargar artefactos entrenados
scaler = joblib.load(os.path.join(BUNDLE_DIR, "scaler.joblib"))
pca_15d = joblib.load(os.path.join(BUNDLE_DIR, "pca_15d.joblib"))
pca_2d = joblib.load(os.path.join(BUNDLE_DIR, "pca_2d.joblib"))
gmm_model = joblib.load(os.path.join(BUNDLE_DIR, "gmm_model.joblib"))
cluster_mapping = joblib.load(os.path.join(BUNDLE_DIR, "cluster_mapping.joblib"))

# Cargar K-Means si está disponible en el bundle
kmeans_model_path = os.path.join(BUNDLE_DIR, "kmeans_model.joblib")
kmeans_mapping_path = os.path.join(BUNDLE_DIR, "kmeans_mapping.joblib")
kmeans_model = joblib.load(kmeans_model_path) if os.path.exists(kmeans_model_path) else None
kmeans_mapping = joblib.load(kmeans_mapping_path) if os.path.exists(kmeans_mapping_path) else cluster_mapping

NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ==============================================================================
# Función Auxiliar: Extracción de 71 Variables Físicas por Segmento
# ==============================================================================
def extract_segment_features(y_seg, sr):
    # A) Separación HPSS local
    y_harm, y_perc = librosa.effects.hpss(y_seg)
    e_harm = float(np.sum(y_harm**2))
    e_perc = float(np.sum(y_perc**2))
    hpss_ratio = (e_perc - e_harm) / (e_perc + e_harm + 1e-9)

    # B) Cromagrama CQT (12 semitonos)
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harm, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma_cqt, axis=1)
    chroma_std = np.std(chroma_cqt, axis=1)

    # C) MFCCs (13 coeficientes)
    mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # D) Contraste Espectral (7 bandas)
    contrast = librosa.feature.spectral_contrast(y=y_seg, sr=sr, n_bands=6)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # E) Descriptores Espectrales
    centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y_seg, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y_seg)

    vec = np.hstack([
        mfcc_mean,                     # 13
        mfcc_std,                      # 13
        chroma_mean,                   # 12
        chroma_std,                    # 12
        contrast_mean,                 # 7
        contrast_std,                  # 7
        np.mean(centroid), np.std(centroid),    # 2
        np.mean(rolloff), np.std(rolloff),      # 2
        np.mean(zcr), np.std(zcr),              # 2
        hpss_ratio                              # 1
    ])
    return vec, chroma_mean

# ==============================================================================
# Rutas y Endpoints
# ==============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze_audio", methods=["POST"])
def analyze_audio():
    if "audio_file" not in request.files:
        return jsonify({"error": "No se subió ningún archivo de audio"}), 400

    file = request.files["audio_file"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    # 1. Cargar señal y recortar silencios extremos
    y, sr = librosa.load(file, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    total_samples = len(y)

    if total_samples < sr * 0.4:
        return jsonify({"error": "El audio es demasiado corto (mínimo 400 ms)"}), 400

    # 2. Separación HPSS global para streaming y descarga
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_path = os.path.join(TEMP_AUDIO_DIR, "current_harmonic.wav")
    perc_path = os.path.join(TEMP_AUDIO_DIR, "current_percussive.wav")
    sf.write(harm_path, y_harm, sr)
    sf.write(perc_path, y_perc, sr)

    # 3. Perfil de notas global con CQT
    chroma_global = librosa.feature.chroma_cqt(y=y_harm, sr=sr, n_chroma=12)
    chroma_vector = np.mean(chroma_global, axis=1)
    indices_notas = np.argsort(chroma_vector)[::-1][:3]
    top_notas = [f"{NOTAS[idx]} ({chroma_vector[idx]:.2f})" for idx in indices_notas]

    # 4. Análisis Temporal Ventana por Ventana (Chunking de 1.5s, solape 0.75s)
    # Evita que el promedio global sature la probabilidad al 100%
    window_size = int(sr * 1.5)
    hop_size = int(sr * 0.75)

    timeline_preds = []

    if total_samples <= window_size:
        # Si el audio dura menos de 1.5s se analiza entero
        feat_chunk, _ = extract_segment_features(y, sr)
        x_scaled = scaler.transform(feat_chunk.reshape(1, -1))
        x_15d = pca_15d.transform(x_scaled)
        pred_c = int(gmm_model.predict(x_15d)[0])
        timeline_preds.append(pred_c)
    else:
        for start in range(0, total_samples - window_size, hop_size):
            chunk = y[start : start + window_size]
            feat_chunk, _ = extract_segment_features(chunk, sr)
            x_scaled = scaler.transform(feat_chunk.reshape(1, -1))
            x_15d = pca_15d.transform(x_scaled)
            pred_c = int(gmm_model.predict(x_15d)[0])
            timeline_preds.append(pred_c)

    # 5. Cálculo de Distribución de Pertenencia Real (% de tiempo activo)
    n_chunks = max(len(timeline_preds), 1)
    counts = {i: timeline_preds.count(i) for i in range(4)}

    # Determinar instrumento predominante en la pista
    predominante_id = max(counts, key=counts.get)
    instrumento_final = cluster_mapping.get(predominante_id, f"Clúster {predominante_id}")
    certidumbre = round((counts[predominante_id] / n_chunks) * 100, 1)

    distribucion = {}
    for c_id in range(4):
        label_name = cluster_mapping.get(c_id, f"Clúster {c_id}")
        pct = round((counts[c_id] / n_chunks) * 100, 1)
        distribucion[f"{label_name} (C{c_id})"] = pct

    # 6. Proyección 2D para visualización del centro de masa de la pista
    feat_global, _ = extract_segment_features(y, sr)
    x_global_scaled = scaler.transform(feat_global.reshape(1, -1))
    x_2d = pca_2d.transform(x_global_scaled)[0]

    return jsonify({
        "instrumento": instrumento_final,
        "cluster_id": predominante_id,
        "certidumbre": certidumbre,
        "distribucion_probabilidad": distribucion,
        "proyeccion_2d": {
            "pc1": round(float(x_2d[0]), 3),
            "pc2": round(float(x_2d[1]), 3)
        },
        "notas_cqt": {
            "labels": NOTAS,
            "valores": [round(float(v), 3) for v in chroma_vector],
            "top_detectadas": top_notas
        }
    })

@app.route("/api/audio_stream/<track_type>")
def audio_stream(track_type):
    fname = "current_harmonic.wav" if track_type == "harmonic" else "current_percussive.wav"
    fpath = os.path.join(TEMP_AUDIO_DIR, fname)
    if not os.path.exists(fpath):
        return jsonify({"error": "Pista no disponible"}), 404

    es_descarga = request.args.get("download", "false").lower() == "true"
    nombre_descarga = "Pista_Armonica.wav" if track_type == "harmonic" else "Pista_Percusiva.wav"

    return send_file(
        fpath,
        mimetype="audio/wav",
        as_attachment=es_descarga,
        download_name=nombre_descarga
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)