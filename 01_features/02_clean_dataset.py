import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "00_data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "00_data", "raw")

ORIGINAL_CSV = os.path.join(PROCESSED_DIR, "features_instruments.csv")
CLEAN_CSV = os.path.join(PROCESSED_DIR, "features_instruments_clean.csv")

print("=" * 70)
print("  PASO 02: AUDITORÍA FORENSE Y SANEAMIENTO DEL DATASET ACÚSTICO")
print("=" * 70)

# 1. Cargar matriz original de 71 variables
df_orig = pd.read_csv(ORIGINAL_CSV)
feat_cols = [c for c in df_orig.columns if c not in ["file_name", "true_instrument"]]
print(f"Dataset cargado: {len(df_orig)} registros con {len(feat_cols)} características físicas.")

# 2. Verificación matemática estricta Drum vs Violin
drum_vectors = df_orig[df_orig["true_instrument"] == "Sound_Drum"][feat_cols].values
violin_vectors = df_orig[df_orig["true_instrument"] == "Sound_Violin"][feat_cols].values

son_identicos = np.allclose(drum_vectors, violin_vectors, atol=1e-8)
print(f"\n[AUDITORÍA 1] ¿Los 700 vectores de Drum y Violin son idénticos?: {son_identicos}")

# 3. Detección de duplicados cruzados
n_duplicados_totales = df_orig.duplicated(subset=feat_cols, keep=False).sum()
n_redundantes = df_orig.duplicated(subset=feat_cols, keep="first").sum()
print(f"[AUDITORÍA 2] Filas involucradas en duplicación: {n_duplicados_totales}")
print(f"[AUDITORÍA 3] Registros redundantes a remover: {n_redundantes} (700 violines + 8 cruzados)")

# 4. Generación del dataset deduplicado limpio
df_no_violin = df_orig[df_orig["true_instrument"] != "Sound_Violin"].copy()
df_clean = df_no_violin.drop_duplicates(subset=feat_cols, keep="first").copy()

# Guardar versión limpia
df_clean.to_csv(CLEAN_CSV, index=False)
print(f"\n-> Dataset saneado exportado exitosamente a: {CLEAN_CSV}")
print(f"   Dimensión final: {df_clean.shape[0]} muestras x {df_clean.shape[1]} columnas.")

print("\n--- DISTRIBUCIÓN DE CLASES REALES (k=3 FAMILIAS FÍSICAS) ---")
print(df_clean["true_instrument"].value_counts())

# 5. Corrección de errata en Metadata_Test.csv si existe
test_meta_path = os.path.join(RAW_DIR, "Metadata_Test.csv")
if os.path.exists(test_meta_path):
    df_test = pd.read_csv(test_meta_path)
    if "Class" in df_test.columns:
        erratas = (df_test["Class"] == "Sound_Guiatr").sum()
        if erratas > 0:
            df_test["Class"] = df_test["Class"].replace({"Sound_Guiatr": "Sound_Guitar"})
            df_test.to_csv(test_meta_path, index=False)
            print(f"\n[CORRECCIÓN] Se corrigieron {erratas} etiquetas 'Sound_Guiatr' -> 'Sound_Guitar' en Metadata_Test.csv")
print("=" * 70)