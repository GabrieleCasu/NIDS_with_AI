import pandas as pd
import sys
import os

# Percorso al file Parquet dei NUOVI ATTACCHI
PARQUET_UNKNOWN_ATTACKS = "../Dataset/NF-UNSW-NB15-V3_ONLY_NEWATTACKS.parquet"

# Nome che il modello si aspetta (CON UNDERSCORE)
NOME_CORRETTO_TARGET = 'NUM_PKTS_256_TO_512_BYTES'

# Nome che il file Parquet aveva erroneamente (CON TRATTINO)
NOME_SCORRETTO_ATTUALE = 'NUM_PKTS_256_TO-512_BYTES'

try:
    print(f"Caricamento del file per la correzione: {PARQUET_UNKNOWN_ATTACKS}")
    df_unknown = pd.read_parquet(PARQUET_UNKNOWN_ATTACKS)

    # Ridenominazione della colonna errata con il nome corretto
    df_unknown.rename(
        columns={NOME_SCORRETTO_ATTUALE: NOME_CORRETTO_TARGET},
        inplace=True,
        errors='raise'  # Se la colonna non esiste, lancia un errore
    )

    print(f"Colonna '{NOME_SCORRETTO_ATTUALE}' rinominata in '{NOME_CORRETTO_TARGET}'.")

    # SOVRASCRIVI IL FILE PARQUET (rendendo la modifica permanente)
    df_unknown.to_parquet(PARQUET_UNKNOWN_ATTACKS, index=False)

    print("\nCorrezione e salvataggio permanente completati.")

except KeyError as e:
    print(f"ERRORE CRITICO: La colonna {e} non è stata trovata nel file.")
except Exception as e:
    print(f"ERRORE: {e}")