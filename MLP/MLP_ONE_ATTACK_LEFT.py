import pandas as pd
from sklearn.metrics import classification_report, roc_curve, \
    auc, accuracy_score
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys  # Per uscire in caso di errore
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Percorso per salvare/caricare il file binario veloce (Parquet)
PARQUET_FILE_PATH = "../Dataset/NF-ToN-IoT-v3_Complete.parquet"

TARGET_COL_BINARIO = "Label"
TARGET_COL_MULTI = "Attack"


#  ARRAY STATICI DEI NOMI DEGLI ASSI
DISPLAY_LABELS_MULTI = np.array(['Backdoor', 'Benign', 'ddos', 'dos', 'injection', 'mitm', 'password',
 'ransomware', 'scanning', 'xss'])
DISPLAY_LABELS_BINARIO = np.array(['Benign', 'Attack'])


# Nomi delle 55 colonne
columns = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "L4_SRC_PORT",
           "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES",
           "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
           "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
           "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
           "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
           "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
           "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
           "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
           "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE", "SRC_TO_DST_IAT_MIN",
           "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_MIN",
           "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV", "Label", "Attack"]


start_time = time.time()
df = None


if os.path.exists(PARQUET_FILE_PATH):
    # SCENARIO 1: CARICAMENTO VELOCE da Parquet
    print(f"Trovato file Parquet. Caricamento veloce da: {PARQUET_FILE_PATH}")
    try:
        df = pd.read_parquet(PARQUET_FILE_PATH)
        load_end_time = time.time()
        print(f"Caricamento da Parquet completato. Tempo: {load_end_time - start_time:.2f} secondi")
    except Exception as e:
        print(f"ERRORE nel caricare Parquet ({e}). Ritorno al caricamento da CSV.")
        df = None

if df is None:
    # SCENARIO 2: CARICAMENTO LENTO - Uscita se Parquet non trovato
    print("ERRORE: Impossibile caricare il DataFrame da CSV o Parquet.")
    sys.exit(1)

# Controlli Finali
print(f"\nDataset Dimensioni finali: {df.shape[0]} righe, {df.shape[1]} colonne")
print(f"Tempo Totale di Caricamento/Pre-processing: {time.time() - start_time:.2f} secondi")


X = df.drop([TARGET_COL_BINARIO, TARGET_COL_MULTI], axis=1, errors="ignore") #il target va messo da parte se no impara la risposta da solo
y = df[TARGET_COL_BINARIO] #colonna di risposta corretta
ATTACK_NAMES_TO_EXCLUDE = DISPLAY_LABELS_MULTI[DISPLAY_LABELS_MULTI != 'Benign']
# Risultato: ['Backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss']

FEATURE_COLUMNS = X.columns.tolist()

cols_to_keep = FEATURE_COLUMNS + [TARGET_COL_BINARIO, TARGET_COL_MULTI]
df = df[cols_to_keep]
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

for col in FEATURE_COLUMNS:
    df[col] = df[col].astype(np.float32)

# Selezioniamo un campione di traffico Benign dal dataframe originale
# che è stato utilizzato nell'addestramento.
BENIGN_SAMPLE_SIZE = 100000
BENIGN_INDEX = np.where(DISPLAY_LABELS_MULTI == 'Benign')[0][0]
df_benign_samples = df[df['Attack'] == BENIGN_INDEX].sample(n=BENIGN_SAMPLE_SIZE, random_state=42)

# X_BENIGN_REF deve usare le stesse feature del training
X_BENIGN_REF = df_benign_samples[FEATURE_COLUMNS].astype(np.float32)
Y_BENIGN_REF = df_benign_samples[TARGET_COL_BINARIO].astype(np.int32)

print(f"\n--- Preparato Campione Benign di Riferimento: {BENIGN_SAMPLE_SIZE} righe ---")

results_list = []
roc_data = {}
#esempi per l'addestramento, poi nella valutazione per ogni attacco verrà presa la quantità intera, escluso benign
MAX_TRAIN_SAMPLES = 10000000



# Inizio del Ciclo di Validazione Leave-One-Attack-Out
print(f"\n--- INIZIO VALIDAZIONE LEAVE-ONE-ATTACK-OUT (9 Cicli) ---")

for excluded_attack in ATTACK_NAMES_TO_EXCLUDE:
    # TROVA L'INDICE NUMERICO DELL'ATTACCO DA ESCLUDERE
    try:
        attack_index_to_exclude = np.where(DISPLAY_LABELS_MULTI == excluded_attack)[0][0]
    except IndexError:
        print(f"Errore: L'attacco '{excluded_attack}' non è in DISPLAY_LABELS_MULTI. Saltato.")
        continue  # Salta se non trova l'indice

    print(f"\n=======================================================")
    print(f"ESCLUSIONE E TEST SU: {excluded_attack} (Indice {attack_index_to_exclude})")

    # CREAZIONE SET DI ADDESTRAMENTO
    # Filtra usando l'indice numerico
    df_train_current = df[df['Attack'] != attack_index_to_exclude] #tutto tranne current

    # CREAZIONE SET DI TEST (100% dell'attacco escluso)

    df_test_current = df[df['Attack'] == attack_index_to_exclude] #solo current

    print(f"Dim. Training: {len(df_train_current):,} | Dim. Test: {len(df_test_current):,}")

    if len(df_test_current) == 0:
        print("ATTENZIONE CRITICA: Dimensione Test Set ancora 0. Controllare allineamento indici.")
        continue
    if len(df_train_current) > MAX_TRAIN_SAMPLES:
        # Usiamo stratify per mantenere le proporzioni tra Benign e gli attacchi rimasti, permette un campionamento fedele al dataset originale
        df_train_current, _ = train_test_split(
            df_train_current,
            train_size=MAX_TRAIN_SAMPLES,
            stratify=df_train_current['Attack'],
            random_state=42
        )

    # PREPARAZIONE X e Y (Con la conversione a 32-bit per stabilità)
    X_train = df_train_current[FEATURE_COLUMNS].astype(np.float32)
    y_train = df_train_current[TARGET_COL_BINARIO].astype(np.int32)

    X_test = df_test_current[FEATURE_COLUMNS].astype(np.float32)
    y_test = df_test_current[TARGET_COL_BINARIO].astype(np.int32)

    # Inizializzazione e Fit dello Scaler
    # L'MLP è estremamente sensibile alla scala. Usiamo lo StandardScaler.
    scaler = StandardScaler(copy=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_BENIGN_REF_scaled = scaler.transform(X_BENIGN_REF)

    # Configurazione dell'MLP
    mlp_bin = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # Due layer: più neuroni del binario perchè il compito si complica
        activation='relu',  # Funzione di attivazione standard
        solver='adam',  # Ottimizzatore più efficiente di SGD
        batch_size= 2048,
        max_iter=20,  # Numero massimo di epoche
        random_state=42,
        verbose=True,  # Mostra il progresso della Loss
        early_stopping=True,  # Ferma il training se la loss non scende più
        n_iter_no_change=5
    )

    # Addestramento
    print("Inizio addestramento MLP...")
    mlp_bin.fit(X_train_scaled, y_train)

    # Predizione
    y_pred = mlp_bin.predict(X_test_scaled)

    # Valutazione
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nReport di Classificazione:")

    #calcolo auc
    #df_test_unito = pd.concat([df_test_current, df_benign_samples])
    #X_test_unito = scaler.transform(df_test_unito[FEATURE_COLUMNS])
    #y_proba_combined = mlp_bin.predict_proba(X_test_unito)[:, 1]

    # COMBINAZIONE DEI DATI per l'AUC
    # Positivi (Attack escluso) + Negativi (Campione Benign)
    #y_true_combined = np.concatenate([y_test, Y_BENIGN_REF])
    y_proba_test = mlp_bin.predict_proba(X_test_scaled)[:, 1]
    y_proba_benign = mlp_bin.predict_proba(X_BENIGN_REF_scaled)[:, 1]

    y_proba_combined = np.concatenate([y_proba_test, y_proba_benign])
    y_true_combined = np.concatenate([y_test, Y_BENIGN_REF])

    #per convertire da probabilità a binario per le metriche
    y_pred_combined = (y_proba_combined >= 0.5).astype(int)
    # Calcolo metriche
    report = classification_report(y_true_combined, y_pred_combined, output_dict=True, zero_division=0)

    # Calcolo AUC sulla Matrice Combinata
    fpr, tpr, thresholds = roc_curve(y_true_combined, y_proba_combined)
    roc_auc = auc(fpr, tpr)

    # Memorizzazione per la curva ROC (usa i dati combinati)
    roc_data[excluded_attack] = {'y_true': y_true_combined, 'y_proba': y_proba_combined, 'auc': roc_auc}

    test_size_current = len(df_test_current)

    # Memorizzazione dei risultati
    results_list.append({
        'attack': excluded_attack,
        'accuracy': report['accuracy'],
        'recall_attack': report['1']['recall'],
        'precision_attack': report['1']['precision'],
        'f1_attack': report['1']['f1-score'],
        'test_size': test_size_current
    })

    print(f"Richiamo su {excluded_attack}: {report['1']['recall']:.4f} | AUC: {roc_auc:.4f} | Precision: {report['1']['precision']:.4f}")

print("\n--- Validazione Leave-One-Attack-Out Completata ---")
df_results = pd.DataFrame(results_list)
# Calcola il peso di ogni attacco (Dimensione test / Totale campioni Attacco)
total_test_samples = df_results['test_size'].sum()
df_results['weight'] = df_results['test_size'] / total_test_samples
print("\n--- RISULTATI MEDI (MEDIA PONDERATA) ---")

weighted_recall = (df_results['recall_attack'] * df_results['weight']).sum()
weighted_precision = (df_results['precision_attack'] * df_results['weight']).sum()
weighted_f1 = (df_results['f1_attack'] * df_results['weight']).sum()

print(f"Richiamo Ponderato: {weighted_recall:.6f}")
print(f"Precisione Ponderata: {weighted_precision:.6f}")
print(f"F1-Score Ponderato: {weighted_f1:.6f}")

# --- PLOTTING DELLE CURVE ROC ---

plt.figure(figsize=(12, 10))
plt.title('Receiver Operating Characteristic (ROC) - Leave-One-Attack-Out')

for attack, data in roc_data.items():
    # Calcola Falsi Positivi (FPR) e Veri Positivi (TPR)
    fpr, tpr, thresholds = roc_curve(data['y_true'], data['y_proba'])
    # Plotting
    plt.plot(fpr, tpr, label=f'ROC {attack} (AUC = {data["auc"]:.4f})')

plt.plot([0, 1], [0, 1], 'r--', label='Casuale (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasso di Falsi Positivi (FPR)')
plt.ylabel('Tasso di Veri Positivi (TPR)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("\n--- Analisi Leave-One-Attack-Out Completata ---")