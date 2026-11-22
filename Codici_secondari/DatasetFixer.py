import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import os
import sys


# Percorso del file CSV esterno
file_path_input = "../Dataset_secondari/NF-UNSW-NB15-v3-CORRECTED_FAST.csv"
# Nuovo file di output FINALIZZATO (Formato Parquet è il più veloce per ML)
file_path_output = "../Dataset/NF-UNSW-NB15-V3_ONLY_NEWATTACKS.parquet"

# Nomi delle 55 colonne (Assumiamo che sia completo e corretto)
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

# Colonne che contengono stringhe e devono essere codificate
COLUMNS_TO_ENCODE = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"]

start_time = time.time()


try:
    print(f"⏳ Caricamento del file CSV da: {file_path_input}")

    # Carica il CSV usando l'intestazione standard e salta la prima riga
    df = pd.read_csv(file_path_input, header=None, skiprows=1, sep=',')
    df.columns = columns

    rows_initial = len(df)

    # --- FILTRO DELLE RIGHE ---
    CLASSES_TO_REMOVE = ['Benign', 'Backdoor', 'DoS']
    df = df[~df['Attack'].astype(str).str.upper().isin([c.upper() for c in CLASSES_TO_REMOVE])].copy()

    print(f"Filtro applicato. Righe rimosse: {rows_initial - len(df)}. Righe finali: {len(df)}")

    # 2. CODIFICA DELLE FEATURE DI STRINGA (LabelEncoder)
    le = LabelEncoder()
    print("Inizio codifica delle feature di stringa...")

    for col in COLUMNS_TO_ENCODE:
        if col in df.columns:
            # Per evitare il crash su NaN, usiamo fillna('MISSING')
            df[col] = le.fit_transform(df[col].fillna('MISSING').astype(str))

    print("Codifica completata.")

    # SALVATAGGIO FINALE IN FORMATO PARQUET
    df.to_parquet(file_path_output, index=False)

    end_time = time.time()
    print(f"\nFile {file_path_output} salvato con successo.")
    print(f"Tempo totale di elaborazione: {end_time - start_time:.2f} secondi")

except Exception as e:
    print(f"\nERRORE CRITICO: {e}")
    sys.exit(1)