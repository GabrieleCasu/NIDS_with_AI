import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys
import seaborn as sns

PARQUET_FILE_PATH = "../Dataset/NF-ToN-IoT-v3_Complete.parquet"
TARGET_COL_BINARIO = "Label"
TARGET_COL_MULTI = "Attack"
DISPLAY_LABELS_MULTI = np.array(['Backdoor', 'Benign', 'ddos', 'dos', 'injection', 'mitm', 'password',
                                 'ransomware', 'scanning', 'xss'])

N_SAMPLES = 10000

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

# Preparazione dati
X = df.drop([TARGET_COL_BINARIO, TARGET_COL_MULTI], axis=1, errors="ignore")
y_multi = df[TARGET_COL_MULTI]
all_features = X.columns

X_filtered = X[all_features].astype(np.float32)

# Normalizzazione (Essenziale per la Cosine Similarity)
print("Normalizzazione delle feature...")
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_filtered)

# Mappa l'array normalizzato ai nomi degli attacchi
df_full = pd.DataFrame(X_normalized)
df_full[TARGET_COL_MULTI] = y_multi.to_numpy()

# Trova l'indice numerico
benign_idx = np.where(DISPLAY_LABELS_MULTI == 'Benign')[0][0] # Risulterà un numero intero

# Estrazione e campionamento dei vettori Benigni
df_benign = df_full[df_full[TARGET_COL_MULTI] == benign_idx]
X_benign_sample = df_benign.sample(n=N_SAMPLES, random_state=42).drop(TARGET_COL_MULTI, axis=1).to_numpy()

# dizionario per salvare le statistiche
similarity_stats = {}

print("\nInizio Calcolo Similarità: Benign vs Attacchi (Vettore-Vettore)...")

# Cicla su tutte le classi di attacco
for attack_name in DISPLAY_LABELS_MULTI:
    if attack_name == 'Benign': #confronta sulla stringa, non serve l'indice
        continue

    attack_idx = np.where(DISPLAY_LABELS_MULTI == attack_name)[0][0]
    df_attack = df_full[df_full[TARGET_COL_MULTI] == attack_idx]
    n_total_attack = len(df_attack)

    # Determina il numero di campioni da prendere, o 5000 o tutti se non c'è ne sono abbastanza
    n_sample_current = min(N_SAMPLES, n_total_attack)

    # Campionamento dell'attacco
    X_attack_sample = df_attack.sample(n=n_sample_current, random_state=42).drop(TARGET_COL_MULTI, axis=1).to_numpy()

    # Calcolo della Similarità del Coseno (Matrice M(benigni) x N(attacchi))
    similarity_matrix = cosine_similarity(X_benign_sample, X_attack_sample)

    # trasforma la matrice in un singolo array di punteggi
    similarity_scores = similarity_matrix.flatten()

    # Calcolo delle Statistiche
    mean = np.mean(similarity_scores)
    std = np.std(similarity_scores)
    min_val = np.min(similarity_scores)
    max_val = np.max(similarity_scores)

    similarity_stats[attack_name] = {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'n_scores': len(similarity_scores)
    }

    print(f"  Statistiche per {attack_name} (n={len(similarity_scores)}): Media={mean:.4f}, Std={std:.4f}")

# Conversione dei risultati in un DataFrame per una facile visualizzazione
stats_df = pd.DataFrame.from_dict(similarity_stats, orient='index')
global_avg = stats_df['mean'].mean()
# grafico
plt.figure(figsize=(12, 7))

# Creazione di una colonna per la Deviazione Standard (da media-std a media+std)
stats_df['mean_minus_std'] = stats_df['mean'] - stats_df['std']
stats_df['mean_plus_std'] = stats_df['mean'] + stats_df['std']

#linee per la similarità
plt.plot(stats_df.index, stats_df['mean'], marker='o', linestyle='none',
         color='royalblue', markersize=8, label='Mean Similarity')

# linee orizzontali per la media
#sns.barplot(x=stats_df.index, y='mean', data=stats_df, color='skyblue', alpha=0.7)
plt.axhline(y=global_avg, color='red', linestyle='--', linewidth=1.5,
            label=f'Global Mean Score ({global_avg:.4f})')

# linee verticali per la Deviazione Standard (Std)
plt.errorbar(x=stats_df.index, y=stats_df['mean'],
             yerr=stats_df['std'],
             fmt='none', capsize=5, color='black', alpha=0.6, label='± 1 Dev. Std.')

# range di Min/Max
for i, attack in enumerate(stats_df.index):
    plt.vlines(i, stats_df.loc[attack, 'min'], stats_df.loc[attack, 'max'],
               color='red', linestyle='--', linewidth=1, label='Min/Max Range' if i == 0 else "")

plt.ylabel('Similarità del Coseno (Benign vs Attacco)')
plt.xlabel('Tipo di Attacco')
plt.ylim(-1.5, 1.5)
plt.title('Similarità (Vettore-Vettore) tra Benign e tutti gli Attacchi')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()