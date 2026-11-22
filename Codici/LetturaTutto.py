import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys  # Per uscire in caso di errore


# Percorso per salvare/caricare il file binario veloce (Parquet)
PARQUET_FILE_PATH = "../Dataset/NF-ToN-IoT-v3_Complete.parquet"
# percorso attacchi sconosciuti
PARQUET_UNKNOWN_ATTACKS = "../Dataset/NF-UNSW-NB15-V3_ONLY_NEWATTACKS.parquet"

TARGET_COL_BINARIO = "Label"
TARGET_COL_MULTI = "Attack"

#limite di scelta fra uno dei flussi noti e sconosciuti, primo test 0.9
THRESHOLD_UNKNOWN = 0.50


#  ARRAY STATICI DEI NOMI DEGLI ASSI
DISPLAY_LABELS_MULTI = np.array(['Backdoor', 'Benign', 'ddos', 'dos', 'injection', 'mitm', 'password',
 'ransomware', 'scanning', 'xss'])
DISPLAY_LABELS_BINARIO = np.array(['Benign', 'Attack'])
df = None
df_unknown = None

# Nomi delle 55 colonne
columns= ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "L4_SRC_PORT",
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

if os.path.exists(PARQUET_FILE_PATH):
    # SCENARIO 1: CARICAMENTO VELOCE da Parquet
    print(f"Trovato file Parquet. Caricamento veloce da: {PARQUET_FILE_PATH}")
    try:
        df = pd.read_parquet(PARQUET_FILE_PATH)
        print(f"Caricamento attacchi sconosciuti")
        df_unknown = pd.read_parquet(PARQUET_UNKNOWN_ATTACKS)
        load_end_time = time.time()
        print(f"Caricamento da Parquet completato. Tempo: {load_end_time - start_time:.2f} secondi")
        df_unknown[TARGET_COL_MULTI] = 10 #non mi interessa avere 7 nuovi attacchi, ma una categoria unica che li comprenda
    except Exception as e:
        print(f"ERRORE nel caricare Parquet ({e}). Ritorno al caricamento da CSV.")
        df = None

if df is None:
    # SCENARIO 2: CARICAMENTO LENTO - Uscita se Parquet non trovato
    print("ERRORE: Impossibile caricare il DataFrame da CSV o Parquet.")
    sys.exit(1)
"""    
pippo = df.columns.tolist()
pluto = df_unknown.columns.tolist()
if pippo == pluto:
    print("ok")
else: print("no")
"""
# Controlli Finali
print(f"\nDataset principale Dimensioni finali: {df.shape[0]} righe, {df.shape[1]} colonne")
print(f"\nDataset secondario dimensioni finali: {df_unknown.shape[0]} righe, {df_unknown.shape[1]} colonne")
print(f"Tempo Totale di Caricamento/Pre-processing: {time.time() - start_time:.2f} secondi")

#feature da aggiungere forse
""""
#creo una nuova feature per aiutare il sistema a distinguere benign e scanning, in quanto hanno importanza abbastanza differente queste 2 feature
#utile solo per binaria
#df['LONGEST_TO_SHORTEST_RATIO'] = (df['LONGEST_FLOW_PKT']/(df['SHORTEST_FLOW_PKT']+1)) #evito divisioni per zero con +1
#print("Aggiunto LONGEST_TO_SHORTEST_RATIO.")
"""
#unione dei dataset
df_combined = pd.concat([df,df_unknown], ignore_index=True) #evita indici duplicati
X = df_combined.drop([TARGET_COL_BINARIO,TARGET_COL_MULTI],axis=1,errors="ignore") #check sulle colonne, anzichè righe
y = df_combined[TARGET_COL_BINARIO]
all_feature_names = X.columns.tolist()
X = X.astype(np.float32)
#-------------------------------------------------------------------
#classificazione binaria
"""
print("\n--- Classificazione binaria ---")
X_train, X_test, y_train, y_test = train_test_split(X[all_feature_names], y, test_size=0.3, random_state=42)
#tolgo valori nulli
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(0, inplace=True)
X_train_32 = X_train.astype(np.float32)
y_train_32 = y_train.astype(np.int32)
x_test_32 = X_test.astype(np.float32)
y_test_32 = y_test.astype(np.int32)

# Random Forest Ottimizzato (n_jobs=7, forse fattibile di più, 10 sembra reggerli da solo, il bilanciamento dà più importanza alla rilevazione di attacchi)
rfc = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1, class_weight="balanced")

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
feature_names = X[all_feature_names].columns
feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)
print("\nFeature in ordine decrescente di importanza(binario): \n")
print(feature_importance_df)
"""
#-----------------------------------------------------------------------------------------
# Classificazione multi-classe+unknown
print("\nClassificazione multi-classe + unknown")
y_multi = df_combined[TARGET_COL_MULTI]
# Train/test split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X[all_feature_names], y_multi, test_size=0.3, random_state=42)

# OTTIMIZZAZIONE MEMORIA: Conversione a 32-bit (Cruciale)
X_train_m.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_m.fillna(0, inplace=True)
X_test_m.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_m.fillna(0, inplace=True)
X_train_m_32 = X_train_m.astype(np.float32)
y_train_m_32 = y_train_m.astype(np.int32)
X_test_m_32 = X_test_m.astype(np.float32)
y_test_m_32 = y_test_m.astype(np.int32)

CUSTOM_WEIGHTS = {8: 3,
                  #1: 2,
                  #6: 5,
                  #2: 2,
                  #7: 5
                  }


# Random Forest (n_jobs=1 per la massima stabilità, probabilmente è aumentabile, con 2 sembra farcela)
# profondità per i primi test 5
rfc_multi = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=2, criterion= 'entropy',class_weight=CUSTOM_WEIGHTS)
#rfc_multi.fit(X_train_m_32, y_train_m_32)
#aumentare la depth per scanning che confonde troppo con benign
#da sistemare gli attacchi con pochi campioni, non vengono rilevati

train_start_time_m = time.time()
print(f"Inizio l'addestramento multi-classe su {X_train_m.shape[0]} righe ...")
rfc_multi.fit(X_train_m_32, y_train_m_32)
train_end_time_m = time.time()

#verifico la probabilità che l'attacco sconosciuto possa essere uno di quelli presenti,
# se è minore del threshold assegna l'indice 10 altrimenti quello della classe con prob maggiore
y_prob_unknown = rfc_multi.predict_proba(X_test_m_32)
max_prob_unknown = np.max(y_prob_unknown, axis=1)
y_pred_multi = np.where(max_prob_unknown<THRESHOLD_UNKNOWN,10,np.argmax(y_prob_unknown, axis=1))


print(f"Tempo di Addestramento (Multi): {train_end_time_m - train_start_time_m:.2f} secondi")
print(f"Accuracy (Multi-classe-test): {accuracy_score(y_test_m.astype(np.int32), y_pred_multi):.4f}")


# Mappa tutti gli indici > 9 al 10 (gli attacchi nuovi).
y_test_m_mapped = np.where(
    y_test_m_32 >= len(DISPLAY_LABELS_MULTI),
    len(DISPLAY_LABELS_MULTI), # Assegna indice 10 (Sconosciuto)
    y_test_m_32
).astype(np.int32)

# Confusion matrix multi-classe
cm_multi = confusion_matrix(y_test_m_mapped, y_pred_multi, normalize="true")
#nuovo indice
labels_matrix = np.arange(len(DISPLAY_LABELS_MULTI)+1)
display_labels = np.append(DISPLAY_LABELS_MULTI,["Novelty"])
#report
print("\nMetriche varie (test-set)")
print(classification_report(y_test_m.astype(np.int32), y_pred_multi, target_names=display_labels, zero_division=0))

# Inizializza Display con i NOMI REALI degli attacchi (Array statico)
disp_multi = ConfusionMatrixDisplay(confusion_matrix=cm_multi, display_labels=display_labels)

# Configurazione Grafica
fig, ax = plt.subplots(figsize=(16, 12))

# PLOT: formato percentuale
disp_multi.plot(cmap="Purples", values_format=".4%", ax=ax)
ax.set_title("Confusion Matrix With Unknown Attacks - FULL Dataset")

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
feature_names = X[all_feature_names].columns
feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)
print("\nFeature in ordine decrescente di importanza:(multi) \n")
print(feature_importance_df)

print("\n--- Analisi Completata ---")

