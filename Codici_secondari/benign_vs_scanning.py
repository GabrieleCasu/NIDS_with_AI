import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import sys

PARQUET_FILE_PATH = "../Dataset/NF-ToN-IoT-v3_Complete.parquet"
TARGET_COL_BINARIO = "Label"
TARGET_COL_MULTI = "Attack"
DISPLAY_LABELS_MULTI = np.array(['Backdoor', 'Benign', 'ddos', 'dos', 'injection', 'mitm', 'password',
                                 'ransomware', 'scanning', 'xss'])
FEATURE_TO_DROP_MULTI = np.array(["FTP_COMMAND_RET_CODE"])

print("Caricamento dataset...")

if not os.path.exists(PARQUET_FILE_PATH):
    print("ERRORE: File Parquet non trovato.")
    sys.exit(1)

try:
    df = pd.read_parquet(PARQUET_FILE_PATH)
    print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
except Exception as e:
    print(f"ERRORE nel caricare Parquet: {e}")
    sys.exit(1)

# Preparazione dati
X = df.drop([TARGET_COL_BINARIO, TARGET_COL_MULTI], axis=1, errors="ignore")
y_multi = df[TARGET_COL_MULTI]
all_features = X.columns

X_filtered = X[all_features].astype(np.float32)

# Normalizzazione (Essenziale per la Cosine Similarity)
print("Normalizzazione delle feature...")
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_filtered)

# calcolo dei centroidi(vettore medio)

def calculate_centroid(X_data, y_labels, target_idx):

    mask = (y_labels == target_idx)
    X_class = X_data[mask]

    if X_class.shape[0] == 0:
        print(f"Errore: Nessun campione trovato per l'indice {target_idx}.")
        return None

    # Restituisce la media delle righe (il vettore centroide)
    return np.mean(X_class, axis=0).reshape(1, -1)

print("ANALISI SIMILARITÀ BENIGN vs. SCANNING ")

# Identificazione degli indici
benign_idx = np.where(DISPLAY_LABELS_MULTI == 'Benign')[0][0]
scanning_idx = np.where(DISPLAY_LABELS_MULTI == 'scanning')[0][0]

y_labels = y_multi.to_numpy()  # Target Label (Indici numerici)

# Calcolo dei Centroidi
print(f"Calcolo centroide Benign ({benign_idx})...")
centroid_benign = calculate_centroid(X_normalized, y_labels, benign_idx)

print(f"Calcolo centroide Scanning ({scanning_idx})...")
centroid_scanning = calculate_centroid(X_normalized, y_labels, scanning_idx)

if centroid_benign is None or centroid_scanning is None:
    sys.exit(1)

# Calcolo della Cosine Similarity
similarity_score = cosine_similarity(centroid_benign, centroid_scanning)[0][0]

# Calcolo dell'Auto-Similarità (per riferimento)
self_sim_benign = cosine_similarity(centroid_benign, centroid_benign)[0][0]
self_sim_scanning = cosine_similarity(centroid_scanning, centroid_scanning)[0][0]

print("\n--- RISULTATI DELLA SIMILARITÀ  ---")

print(f"\nBENIGN vs SCANNING (Similarità Media): {similarity_score:.6f}")
print(f"BENIGN vs BENIGN (Auto-Similarità): {self_sim_benign:.6f}")
print(f"SCANNING vs SCANNING (Auto-Similarità): {self_sim_scanning:.6f}")
