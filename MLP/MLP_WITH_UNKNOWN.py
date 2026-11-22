import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# --- CONFIGURAZIONE ---
PARQUET_FILE_PATH = "../Dataset/NF-ToN-IoT-v3_Complete.parquet"
PARQUET_UNKNOWN_ATTACKS = "../Dataset/NF-UNSW-NB15-V3_ONLY_NEWATTACKS.parquet"
TARGET_COL_BINARIO = "Label"
TARGET_COL_MULTI = "Attack"
THRESHOLD_UNKNOWN = 0.50 

N_TARGET_SAMPLES = 10000000  # Dimensione desiderata per il dataset noto
CLASSI_INTERE = ['mitm', 'ransomware']
DISPLAY_LABELS_MULTI = np.array(['Backdoor', 'Benign', 'ddos', 'dos', 'injection', 'mitm', 'password',
                                 'ransomware', 'scanning', 'xss'])
DISPLAY_LABELS_WITH_unknown = np.append(DISPLAY_LABELS_MULTI, ["unknown"])

start_time = time.time()

# --- CARICAMENTO ---
print("Caricamento dataset...")
df = pd.read_parquet(PARQUET_FILE_PATH)
df_unknown = pd.read_parquet(PARQUET_UNKNOWN_ATTACKS)
df_unknown[TARGET_COL_MULTI] = 10 # Etichetta per attacchi ignoti

# --- CAMPIONAMENTO PROPORZIONALE ---
print("\nInizio campionamento proporzionale...")

# Identifichiamo gli indici numerici delle classi da recuperare interamente
idx_intere = [np.where(DISPLAY_LABELS_MULTI == c)[0][0] for c in CLASSI_INTERE]

# Separiamo le classi "intere" dal resto per calcolare le proporzioni sul rimanente
df_rare = df[df[TARGET_COL_MULTI].isin(idx_intere)]
df_others = df[~df[TARGET_COL_MULTI].isin(idx_intere)]

n_rare = len(df_rare)
n_to_sample_others = N_TARGET_SAMPLES - n_rare

# Campionamento stratificato sul resto del dataset (mantiene le proporzioni originali)
# Usiamo train_test_split come trucco veloce per lo stratified sampling
df_others_sampled, _ = train_test_split(
    df_others, 
    train_size=n_to_sample_others, 
    stratify=df_others[TARGET_COL_MULTI], 
    random_state=42
)

# Unione finale: Proporzionali + Rare Full + Unknown Full
df_combined = pd.concat([df_others_sampled, df_rare, df_unknown], ignore_index=True)

print(f"Riepilogo Dataset:")
print(f" - Classi proporzionali: {len(df_others_sampled)}")
print(f" - Classi rare (100%): {n_rare}")
print(f" - Attacchi ignoti: {len(df_unknown)}")
print(f"Totale righe: {len(df_combined)}")

# --- PRE-PROCESSING ---
X = df_combined.drop([TARGET_COL_BINARIO, TARGET_COL_MULTI], axis=1).astype(np.float32)
y_multi = df_combined[TARGET_COL_MULTI].astype(np.int32)

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Split Training/Test
X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.3, random_state=42, stratify=y_multi)

# Scaling (Essenziale per MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- ADDESTRAMENTO MLP ---
print(f"\nAddestramento MLP su {X_train.shape[0]} campioni...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=100,
    early_stopping=True,
    verbose=True,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# --- unknown DETECTION ---
print(f"\nAnalisi probabilità (Soglia: {THRESHOLD_UNKNOWN})...")
y_probs = mlp.predict_proba(X_test_scaled)
max_probs = np.max(y_probs, axis=1)

# Logica: se confidenza < soglia -> unknown (10) STESSO METODO DI RF
y_pred_multi = np.where(max_probs < THRESHOLD_UNKNOWN, 10, np.argmax(y_probs, axis=1))
y_test_mapped = np.where(y_test >= 10, 10, y_test)

# --- RISULTATI ---
print("\n" + "="*30)
print("REPORT FINALE - MLP unknown DETECTION")
print("="*30)
print(f"Accuracy: {accuracy_score(y_test, y_pred_multi):.4f}")
print(classification_report(y_test_mapped, y_pred_multi, target_names=DISPLAY_LABELS_WITH_unknown, digits=4))

# Matrice di Confusione
cm = confusion_matrix(y_test_mapped, y_pred_multi, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS_WITH_unknown)

fig, ax = plt.subplots(figsize=(14, 10))
disp.plot(cmap="Purples", values_format=".2%", ax=ax)
ax.set_title(f"MLP: Known vs Novel Attacks (Proportional Sampling)\nThreshold: {THRESHOLD_UNKNOWN}")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print(f"\nTempo totale: {time.time() - start_time:.2f}s")