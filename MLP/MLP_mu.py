import os
import sys
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

PARQUET_FILE_PATH = "../Dataset/NF-ToN-IoT-v3_Complete.parquet"
TARGET_COL_BINARIO = "Label"
TARGET_COL_MULTI = "Attack"
DISPLAY_LABELS_BINARIO = np.array(['Benign', 'Attack'])
DISPLAY_LABELS_MULTI = np.array(['Backdoor', 'Benign', 'ddos', 'dos', 'injection', 'mitm', 'password',
                                 'ransomware', 'scanning', 'xss'])
CLASSI_INTERE = ['mitm','ransomware', 'Backdoor', 'ddos', 'dos', 'injection', 'password','xss','scanning']

print("Caricamento dataset...")
start_time = time.time()

if not os.path.exists(PARQUET_FILE_PATH):
    print("ERRORE: File Parquet non trovato.")
    sys.exit(1)

try:
    df = pd.read_parquet(PARQUET_FILE_PATH)
    print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
except Exception as e:
    print(f"ERRORE nel caricare Parquet: {e}")
    sys.exit(1)

# Campionamento prima dello split
#df_sampled = df.sample(n=3000000, random_state=42)
N_TARGET_TOTAL = 10000000
# Calcoliamo prima quanti esempi hanno MITM e Ransomware in totale nel dataset originale
# Usiamo gli indici numerici corretti basati sulla posizione in DISPLAY_LABELS_MULTI
idx_mitm = np.where(DISPLAY_LABELS_MULTI == 'mitm')[0][0]
idx_ransomware = np.where(DISPLAY_LABELS_MULTI == 'ransomware')[0][0]

df_mitm_full = df[df[TARGET_COL_MULTI] == idx_mitm]
df_ransom_full = df[df[TARGET_COL_MULTI] == idx_ransomware]

n_mitm = len(df_mitm_full)
n_ransom = len(df_ransom_full)

# Sottraiamo questi dal totale per sapere quanto campionare dalle altre classi
n_others_to_sample = N_TARGET_TOTAL - (n_mitm + n_ransom)

# Creiamo il dataset filtrando via le due classi intere per campionare il resto
df_others = df[~df[TARGET_COL_MULTI].isin([idx_mitm, idx_ransomware])]

# Campionamento stratificato sul resto del dataset
df_others_sampled = df_others.sample(n=n_others_to_sample, random_state=42)

# Uniamo i pezzi: Resto Campionato + MITM Full + Ransomware Full
df_sampled = pd.concat([df_others_sampled, df_mitm_full, df_ransom_full])

print(f"\n--- Riepilogo Campionamento ---")
print(f"Altre classi (proporzionali): {len(df_others_sampled)}")
print(f"MITM (Recupero totale): {n_mitm}")
print(f"Ransomware (Recupero totale): {n_ransom}")
print(f"Totale dataset: {len(df_sampled)}")

# Verifica finale della distribuzione ID
print("\nDistribuzione finale nel set di training:")
print(df_sampled[TARGET_COL_MULTI].value_counts().sort_index())

"""
sampled_frames = []

print("Inizio campionamento mirato...")

for i, attack_name in enumerate(DISPLAY_LABELS_MULTI):
    # FILTRO CORRETTO: usiamo l'indice i che corrisponde alla posizione nel DISPLAY_LABELS_MULTI
    df_class = df[df[TARGET_COL_MULTI] == i]
    n_available = len(df_class)

    # Il controllo va fatto comunque sul nome stringa
    if attack_name in CLASSI_INTERE:
        if n_available > 0:
            sampled_frames.append(df_class)
            print(f" - {attack_name} (ID {i}): Recuperati TUTTI gli esempi ({n_available})")
        else:
            print(f" - ATTENZIONE: {attack_name} (ID {i}) non trovato nel dataset!")
    else:
        if n_available > 0:
            n_to_sample = min(max_samples_others, n_available)
            df_sub = df_class.sample(n=n_to_sample, random_state=42)
            sampled_frames.append(df_sub)
            print(f" - {attack_name} (ID {i}): Campionati {n_to_sample} su {n_available}")

# Uniamo tutto nel nuovo dataset di training
df_sampled = pd.concat(sampled_frames)

print(f"\nNuovo dataset pronto: {len(df_sampled)} righe totali.")
"""
X = df_sampled.drop(['Label', 'Attack'], axis=1)
y = df_sampled['Attack']

# Split dei dati (Importante per valutare il modello su dati mai visti)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Inizializzazione e Fit dello Scaler
# L'MLP è estremamente sensibile alla scala. Usiamo lo StandardScaler.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurazione dell'MLP
mlp_bin = MLPClassifier(
    hidden_layer_sizes=(128, 64), # Due layer: più neuroni del binario perchè il compito si complica
    activation='relu',           # Funzione di attivazione standard
    solver='adam',               # Ottimizzatore più efficiente di SGD
    max_iter=100,                # Numero massimo di epoche
    random_state=42,
    verbose=True,                # Mostra il progresso della Loss
    early_stopping=True          # Ferma il training se la loss non scende più
)

# Addestramento
print("Inizio addestramento MLP...")
mlp_bin.fit(X_train_scaled, y_train)

# Predizione
y_pred = mlp_bin.predict(X_test_scaled)

# Valutazione
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nReport di Classificazione:")
print(classification_report(y_test, y_pred, target_names=DISPLAY_LABELS_MULTI, digits=4))
print("\nMatrice di Confusione:")
cm = confusion_matrix(y_test.astype(np.int32), y_pred, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS_MULTI)
fig, ax = plt.subplots(figsize=(16, 12))

# PLOT: formato percentuale
disp.plot(cmap="Purples", values_format=".4%", ax=ax)
ax.set_title("Confusion Matrix With MLP - 2M examples")

# Riduzione Forzata del Font nelle Celle, altrimenti i valori sforano
for row in disp.text_:
    for text in row:
        text.set_fontsize(7) # Font 7 per adattare i numeri esponenziali

plt.xticks(rotation=90) # Ruota i nomi X
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()