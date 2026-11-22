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
PARQUET_UNKNOWN_ATTACKS = "../Dataset/NF-UNSW-NB15-V3_ONLY_NEWATTACKS.parquet"



print("Caricamento dataset...")
start_time = time.time()

if not os.path.exists(PARQUET_FILE_PATH):
    print("ERRORE: File Parquet non trovato.")
    sys.exit(1)

try:
    df = pd.read_parquet(PARQUET_FILE_PATH)
    print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    df_unknown = pd.read_parquet(PARQUET_UNKNOWN_ATTACKS)
    print(f"Dataset secondario caricato: {df_unknown.shape[0]} righe, {df_unknown.shape[1]} colonne")
except Exception as e:
    print(f"ERRORE nel caricare Parquet: {e}")
    sys.exit(1)

df_sampled = df.sample(n=10000000, random_state=42)

X = df_sampled.drop(['Label', 'Attack'], axis=1)
y = df_sampled['Label']

# Split dei dati (Importante per valutare il modello su dati mai visti)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
#il test invece include i nuovi attacchi
X_test_combined = pd.concat([X_test, df_unknown.drop(['Label', 'Attack'], axis=1, errors='ignore')])
y_test_combined = pd.concat([y_test, df_unknown['Label']]).astype(np.int32)

X_test_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_combined.fillna(0, inplace=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test_combined)

# Configurazione dell'MLP
mlp_bin = MLPClassifier(
    hidden_layer_sizes=(64, 32), # Due layer: uno da 64 e uno da 32 neuroni
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
print(f"Accuracy: {accuracy_score(y_test_combined, y_pred):.4f}")
print("\nReport di Classificazione:")
print(classification_report(y_test_combined, y_pred, target_names=DISPLAY_LABELS_BINARIO))
print("\nMatrice di Confusione:")
cm = confusion_matrix(y_test_combined.astype(np.int32), y_pred, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS_BINARIO)
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format=".4%")
plt.title("Confusion Matrix (Binaria)")
plt.show()