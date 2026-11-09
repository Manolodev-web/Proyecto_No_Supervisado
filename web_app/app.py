import os
import sys
import glob
import time
import uuid
import joblib
import librosa
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

# Asegurar importación de audio_features sin importar la ruta de ejecución
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, os.path.join(PROJECT_ROOT, "01_features")]:
    if p not in sys.path:
        sys.path.append(p)

from audio_features import extract_71_audio_features, NOTAS_CQT

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # Límite máximo: 15 MB
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac"}

# Configuración CORS: Permite peticiones desde tu sitio en Netlify y en local
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://dat255.netlify.app",
            "http://localhost:*",
            "http://127.0.0.1:*"
        ]
    }
})

BUNDLE_DIR = os.path.join(CURRENT_DIR, "models_bundle")
TEMP_AUDIO_DIR = os.path.join(CURRENT_DIR, "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Cargar artefactos preentrenados en memoria al arranque
scaler = joblib.load(os.path.join(BUNDLE_DIR, "scaler.joblib"))
pca_15d = joblib.load(os.path.join(BUNDLE_DIR, "pca_15d.joblib"))
pca_2d = joblib.load(os.path.join(BUNDLE_DIR, "pca_2d.joblib"))
gmm_model = joblib.load(os.path.join(BUNDLE_DIR, "gmm_model.joblib"))
kmeans_model = joblib.load(os.path.join(BUNDLE_DIR, "kmeans_model.joblib"))
cluster_mapping = joblib.load(os.path.join(BUNDLE_DIR, "cluster_mapping.joblib"))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def purge_old_temp_files(max_age_seconds=600):
    """Limpia archivos de audio temporales que tengan más de 10 minutos."""
    now = time.time()
    for f in glob.glob(os.path.join(TEMP_AUDIO_DIR, "*.wav")):
        try:
            if os.stat(f).st_mtime < now - max_age_seconds:
                os.remove(f)
        except OSError:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Endpoint ligero para verificar que el servicio esté despierto."""
    return jsonify({"status": "ok", "service": "audio-clustering-api"}), 200


@app.route("/api/analyze_audio", methods=["POST"])
def analyze_audio():
    purge_old_temp_files()

    if "audio_file" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo de audio"}), 400

    file = request.files["audio_file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Formato de archivo no válido (Use .wav, .mp3 o .ogg)"}), 400

    try:
        y, sr = librosa.load(file, sr=22050, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)
    except Exception as e:
        return jsonify({"error": f"Error decodificando audio: {str(e)}"}), 400

    total_samples = len(y)
    if total_samples < sr * 0.3:
        return jsonify({"error": "Audio demasiado breve (mínimo 300 ms)"}), 400

    # 1. Extracción física canónica de 71 dimensiones
    feats_global, chroma_global, y_harm, y_perc = extract_71_audio_features(y, sr)

    # 2. Guardar pistas temporales protegidas por sesión UUID
    session_id = str(uuid.uuid4())
    harm_filename = f"{session_id}_harmonic.wav"
    perc_filename = f"{session_id}_percussive.wav"
    sf.write(os.path.join(TEMP_AUDIO_DIR, harm_filename), y_harm, sr)
    sf.write(os.path.join(TEMP_AUDIO_DIR, perc_filename), y_perc, sr)

    # 3. Inferencia acústica en espacio latente unificado (15D PCA)
    x_scaled = scaler.transform(feats_global.reshape(1, -1))
    x_15d = pca_15d.transform(x_scaled)
    x_2d = pca_2d.transform(x_scaled)[0]

    probs_gmm = gmm_model.predict_proba(x_15d)[0]
    c_gmm = int(np.argmax(probs_gmm))
    c_km = int(kmeans_model.predict(x_15d)[0])
    inst_pred = cluster_mapping.get(c_gmm, f"Clúster {c_gmm}")
    confianza = round(float(np.max(probs_gmm)) * 100, 2)

    # 4. Análisis temporal continuo por ventanas
    window_size = int(sr * 1.5)
    hop_size = int(sr * 0.75)
    timeline_clusters = []

    if total_samples <= window_size:
        timeline_clusters.append(c_gmm)
    else:
        starts = list(range(0, total_samples - window_size, hop_size))
        if starts[-1] + window_size < total_samples:
            starts.append(total_samples - window_size)

        for st in starts:
            chunk = y[st : st + window_size]
            c_feat, _, _, _ = extract_71_audio_features(chunk, sr)
            c_15d = pca_15d.transform(scaler.transform(c_feat.reshape(1, -1)))
            timeline_clusters.append(int(gmm_model.predict(c_15d)[0]))

    n_chunks = len(timeline_clusters)
    distribucion = {}
    for c_id in sorted(cluster_mapping.keys()):
        inst_name = cluster_mapping[c_id]
        pct = round((timeline_clusters.count(c_id) / n_chunks) * 100, 1)
        distribucion[f"{inst_name} (C{c_id})"] = pct

    # Detección tonal CQT
    top_idxs = np.argsort(chroma_global)[::-1][:3]
    top_notas = [f"{NOTAS_CQT[i]} ({chroma_global[i]:.2f})" for i in top_idxs]

    return jsonify({
        "session_id": session_id,
        "instrumento": inst_pred,
        "cluster_id": c_gmm,
        "cluster_kmeans": c_km,
        "certidumbre": confianza,
        "distribucion_probabilidad": distribucion,
        "proyeccion_2d": {
            "pc1": round(float(x_2d[0]), 3),
            "pc2": round(float(x_2d[1]), 3)
        },
        "notas_cqt": {
            "labels": NOTAS_CQT,
            "valores": [round(float(v), 3) for v in chroma_global],
            "top_detectadas": top_notas
        }
    })


@app.route("/api/audio_stream/<session_id>/<track_type>")
def audio_stream(session_id, track_type):
    clean_id = "".join([c for c in session_id if c.isalnum() or c == "-"])
    fname = f"{clean_id}_harmonic.wav" if track_type == "harmonic" else f"{clean_id}_percussive.wav"
    fpath = os.path.join(TEMP_AUDIO_DIR, fname)

    if not os.path.exists(fpath):
        return jsonify({"error": "Pista expirada o no encontrada"}), 404

    es_descarga = request.args.get("download", "false").lower() == "true"
    nombre_descarga = "Pista_Armonica.wav" if track_type == "harmonic" else "Pista_Percusiva.wav"

    return send_file(fpath, mimetype="audio/wav", as_attachment=es_descarga, download_name=nombre_descarga)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)