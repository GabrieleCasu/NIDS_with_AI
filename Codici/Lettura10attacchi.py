import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys  # Per uscire in caso di errore

# Percorso al file CSV CORRETTO e COMPLETO (usato solo la prima volta)
CSV_FILE_PATH = "../Dataset_secondari/NF-ToN-IoT-v3_CORRECTED_FAST.csv"
# Percorso per salvare/caricare il file binario veloce (Parquet)
PARQUET_FILE_PATH = "../Dataset/NF-ToN-IoT-v3_Complete.parquet"

TARGET_COL_BINARIO = "Label"
TARGET_COL_MULTI = "Attack"
#CATEGORICAL_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"] codificate per permettere l'addestramento
#la codifica effettuata prima di convertirlo in parquet era questa, ordina gli attacchi alfabeticamente da 0 a 9 e
# gli ip dal primo, per esempio 1.1.1.1, sempre con numeri da 0 a seguire
"""
le = LabelEncoder()
for col in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"]:
    df[col] = le.fit_transform(df[col].astype(str))
"""

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

FEATURE_TO_DROP_BINARIO = np.array(["RETRANSMITTED_OUT_PKTS", "RETRANSMITTED_OUT_BYTES"]) #importanza nulla, toglierli aumenta leggermente l'accuratezza del benigno e molto quella dell'attacco
FEATURE_TO_DROP_MULTI = np.array(["FTP_COMMAND_RET_CODE"]) #relativo al protocollo ftp, toglierlo danneggia mitm(5%), ma aumenta molto ramsomware(9%), gli altri subiscono variazioni minime

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

#creo una nuova feature per aiutare il sistema a distinguere benign e scanning, in quanto hanno importanza abbastanza differente queste 2 feature
#utile solo per binaria
#df['LONGEST_TO_SHORTEST_RATIO'] = (df['LONGEST_FLOW_PKT']/(df['SHORTEST_FLOW_PKT']+1)) #evito divisioni per zero con +1
#print("Aggiunto LONGEST_TO_SHORTEST_RATIO.")

# soglia per le porte note
THRESHOLD_WELL_KNOWN_PORT = 1024

# 1 se la porta è < 1024, 0 altrimenti
df['IS_KNOWN_SERVICE_PORT'] = np.where(df['L4_DST_PORT'] <= THRESHOLD_WELL_KNOWN_PORT, 1, 0)
print("Aggiunto IS_KNOWN_SERVICE_PORT.")

#utile solo per il multi
df['DURATION_TTL_RATIO'] = (    df['FLOW_DURATION_MILLISECONDS'] / (df['MIN_TTL'] + 1))
print("Aggiunto DURATION_TTL_RATIO.")

CUSTOM_WEIGHTS = {8: 3,#scanning, perchè ha difficoltà
                  1: 1,#benigno
                  5: 5,#mitm, perchè sono pochi campioni
                  7: 5}#ransomware, perchè sono pochi campioni

X = df.drop([TARGET_COL_BINARIO, TARGET_COL_MULTI], axis=1, errors="ignore") #il target va messo da parte se no impara la risposta da solo
y = df[TARGET_COL_BINARIO] #colonna di risposta corretta
all_feature_names = X.columns.tolist()

 # CLASSIFICAZIONE BINARIA (Target: Label)
"""
print("\n--- 3. Classificazione Binaria (Label) ---")


# Usa una list comprehension per escludere le due colonne
FEATURES_FOR_BINARY = [
    col for col in all_feature_names if col not in FEATURE_TO_DROP_BINARIO
]
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X[FEATURES_FOR_BINARY], y, test_size=0.3, random_state=42)

# OTTIMIZZAZIONE MEMORIA: Conversione a 32-bit (Cruciale per la stabilità)
X_train_32 = X_train.astype(np.float32)
y_train_32 = y_train.astype(np.int32)
x_test_32 = X_test.astype(np.float32)
y_test_32 = y_test.astype(np.int32)

# Random Forest Ottimizzato (n_jobs=7, forse fattibile di più, 10 sembra reggerli da solo, il bilanciamento dà più importanza alla rilevazione di attacchi)
rfc = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=8, class_weight="balanced")

train_start_time = time.time()
print(f"Inizio l'addestramento binario su {X_train.shape[0]} righe...")
rfc.fit(X_train_32, y_train_32) #crea la foresta
train_end_time = time.time()

y_pred = rfc.predict(X_test.astype(np.float32)) #previsioni su cui basa le statistiche

print(f"Tempo di Addestramento (Binario): {train_end_time - train_start_time:.2f} secondi")
print(f"Accuracy (Binaria-test): {accuracy_score(y_test.astype(np.int32), y_pred):.4f}")

print("\nMetriche varie-test")
print(classification_report(y_test.astype(np.int32), y_pred, target_names=DISPLAY_LABELS_BINARIO, zero_division=0))
cm = confusion_matrix(y_test.astype(np.int32), y_pred, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS_BINARIO)
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format=".2%")
plt.title("Confusion Matrix (Binaria)")
plt.show()

print("\nFeature importance")
importance = rfc.feature_importances_
feature_names = X[FEATURES_FOR_BINARY].columns
feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)
print("\nFeature in ordine decrescente di importanza(binario): \n")
print(feature_importance_df)
"""
# CLASSIFICAZIONE MULTI-CLASSE (Target: Attack)

print("\n--- 4. Classificazione Multi-Classe (Attack) ---")
y_multi = df[TARGET_COL_MULTI]
FEATURES_FOR_MULTI = [
    col for col in all_feature_names if col not in FEATURE_TO_DROP_MULTI
]

# Train/test split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X[FEATURES_FOR_MULTI], y_multi, test_size=0.3, random_state=42)

# OTTIMIZZAZIONE MEMORIA: Conversione a 32-bit (Cruciale)
X_train_m_32 = X_train_m.astype(np.float32)
y_train_m_32 = y_train_m.astype(np.int32)
X_test_m_32 = X_test_m.astype(np.float32)
y_test_m_32 = y_test_m.astype(np.int32)

# Random Forest (n_jobs=1 per la massima stabilità, probabilmente è aumentabile, con 2 sembra farcela)
# profondità per i primi test 5
rfc_multi = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=2, criterion= 'entropy',class_weight=CUSTOM_WEIGHTS)
#rfc_multi.fit(X_train_m_32, y_train_m_32)
#aumentare la depth per scanning che confonde troppo con benign
#da sistemare gli attacchi con pochi campioni, non vengono rilevati

train_start_time_m = time.time()
print(f"Inizio l'addestramento multi-classe su {X_train_m.shape[0]} righe ...")
rfc_multi.fit(X_train_m_32, y_train_m_32) #addestramento
train_end_time_m = time.time()

y_pred_multi = rfc_multi.predict(X_test_m.astype(np.float32))

print(f"Tempo di Addestramento (Multi): {train_end_time_m - train_start_time_m:.2f} secondi")
print(f"Accuracy (Multi-classe-test): {accuracy_score(y_test_m.astype(np.int32), y_pred_multi):.4f}")
print("\nMetriche varie (test-set)")
print(classification_report(y_test_m.astype(np.int32), y_pred_multi, target_names=DISPLAY_LABELS_MULTI, zero_division=0))


# Confusion matrix multi-classe
cm_multi = confusion_matrix(y_test_m.astype(np.int32), y_pred_multi, normalize="true")

# Inizializza Display con i NOMI REALI degli attacchi (Array statico)
disp_multi = ConfusionMatrixDisplay(confusion_matrix=cm_multi, display_labels=DISPLAY_LABELS_MULTI)

# Configurazione Grafica
fig, ax = plt.subplots(figsize=(16, 12))

# PLOT: formato percentuale
disp_multi.plot(cmap="Purples", values_format=".4%", ax=ax)
ax.set_title("Confusion Matrix - FULL Dataset")

# Riduzione Forzata del Font nelle Celle, altrimenti i valori sforano
for row in disp_multi.text_:
    for text in row:
        text.set_fontsize(7) # Font 7 per adattare i numeri esponenziali

plt.xticks(rotation=90) # Ruota i nomi X
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nFeature importance")
importance = rfc_multi.feature_importances_
feature_names = X[FEATURES_FOR_MULTI].columns
feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)
print("\nFeature in ordine decrescente di importanza:(multi) \n")
print(feature_importance_df)

print("\n--- Analisi Completata ---")
