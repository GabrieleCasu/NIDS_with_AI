import pandas as pd
import numpy as np
import time
import sys

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PARQUET_FILE_PATH = "../Dataset/NF-UNSW-NB15-V3_ONLY_NEWATTACKS.parquet"

# Array dei nomi degli attacchi (necessario per il loop e l'output)
DISPLAY_LABELS_MULTI = np.array(['Analysis','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms'])
#FEATURE_TO_DROP_BACKDOOR_DOS_INJECTION_MITM_RAMSOMWARE_XSS = ["FTP_COMMAND_RET_CODE"] #capire come levarlo a questi attacchi e basta
#ATTACKS_WITHOUT_ONE_FEATURE = ['Backdoor', 'dos', 'injection', 'mitm', 'ramsomware', 'xss']

# Nomi delle 55 colonne (Necessario per lo split)
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

TARGET_COL_BINARIO = "Label"
TARGET_COL_MULTI = "Attack"
start_time = time.time()
df = None

# Caricamento del DataFrame dal Parquet (Assumiamo che il file sia già pre-elaborato)
try:
    df = pd.read_parquet(PARQUET_FILE_PATH)
    load_end_time = time.time()
    print(f"Caricamento da Parquet completato. Tempo: {load_end_time - start_time:.2f} secondi")

except Exception as e:
    print(f"ERRORE: Impossibile caricare il DataFrame da Parquet: {e}")
    sys.exit(1)

# PREPARAZIONE DEI DATI OVR (Split deterministico)
print("Nomi delle colonne nel DataFrame 'df':")
print(df.columns.tolist())
# PULIZIA DI INFINITI

print("Inizio pulizia dei valori Infiniti...")

# Sostituisci tutti gli infiniti con NaN
df.replace([float('inf'), -float('inf')], np.nan, inplace=True)

# Riempi tutti i NaN rimanenti con 0 (per preparare la conversione)
df.fillna(0, inplace=True)

print("Pulizia completata.")

# Definisci i set di feature e target
X = df.drop([TARGET_COL_BINARIO, TARGET_COL_MULTI], axis=1, errors="ignore")
y_multi = df[TARGET_COL_MULTI]

# Split deterministico (random_state=42) per ottenere X_train e y_train
# Usiamo solo i dati di training (70%) per l'analisi di importanza
X_train_m, _, y_train_m, _ = train_test_split(X, y_multi, test_size=0.3, random_state=42)

# OTTIMIZZAZIONE MEMORIA: Conversione a 32-bit (Cruciale)
X_train_m_32 = X_train_m.astype(np.float32)
y_train_m_32 = y_train_m.astype(np.int32)
feature_names = X.columns  # Le feature sono le colonne del DataFrame X


# ANALISI FEATURE IMPORTANCE PER CLASSE (OvR(One vs Rest) LOOP)

print("\n\n=== ANALISI FEATURE IMPORTANCE PER-CLASSE (OvR LOOP) ===")
start_ovr_time = time.time()
all_feature_ranks = {}
#X_reduced = X.drop(FEATURE_TO_DROP_BACKDOOR_DOS_INJECTION_MITM_RAMSOMWARE_XSS)
#X_reduced = X_reduced.astype(np.float32)

for i, attack_name in enumerate(DISPLAY_LABELS_MULTI):
    print(f"\n--- Addestramento OvR per: {attack_name} ({i + 1}/7) ---")

    # Crea il target binario (OvR: La classe attuale è 1, tutte le altre sono 0)
    y_ovr_train = np.where(y_train_m_32 == i, 1, 0)

    # Addestra il Modello Binario specifico per questa minaccia
    rfc_ovr = RandomForestClassifier(n_estimators=50,max_depth=15,random_state=42,n_jobs=2,criterion='entropy')

    try:
        rfc_ovr.fit(X_train_m_32, y_ovr_train)

        # Estrai l'Importanza delle Feature e ordina
        importances = rfc_ovr.feature_importances_
        df_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

        # Salva e Stampa i risultati
        all_feature_ranks[attack_name] = df_importance

        print(f"Classifica completata. {attack_name}:")
        print(df_importance.to_string(index=False))


    except Exception as e:
        print(f"ERRORE GRAVE DURANTE L'ADDESTRAMENTO OVR per {attack_name}: {e}")
        print("Interruzione del ciclo.")
        break

print("\n=== ANALISI OVR COMPLETATA ===")
print(f"Tempo totale per l'analisi OvR: {time.time() - start_ovr_time:.2f} secondi")