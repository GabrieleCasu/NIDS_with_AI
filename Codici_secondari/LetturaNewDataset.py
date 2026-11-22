import pandas as pd
import time
import os
import sys  # Per uscire in caso di errore

# Percorso per salvare/caricare il file binario veloce (Parquet)
PARQUET_FILE_PATH = "../Dataset/NF-UNSW-NB15-V3_ONLY_NEWATTACKS.parquet"

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

# Controlli Finali, le righe con flussi 'Benign','Backdoor' e 'DoS sono state rimosse in quanto già presenti nel dataset originale
print(f"\nDataset Dimensioni finali: {df.shape[0]} righe, {df.shape[1]} colonne")
print(f"Tempo Totale di Caricamento/Pre-processing: {time.time() - start_time:.2f} secondi")
print("\n--- DEBUG: Nomi delle Colonne nel Dataset Esterno ---")
print(df.columns.tolist())
print("-----------------------------------------------------")

print("\n--- Conteggio delle Classi di Attacco nel Dataset Esterno ---")
try:
    # conteggio sull'ULTIMA COLONNA del DataFrame
    label_column_name = df.columns[-1]
    attack_counts = df[label_column_name].value_counts()

    print(f"Risultati per la colonna '{label_column_name}':")
    print(attack_counts.to_string())
except Exception as e:
    print(f"ERRORE: Impossibile eseguire value_counts. {e}")
