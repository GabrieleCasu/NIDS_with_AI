import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys

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
""" per tornare ai centroidi decommenta
print("\n--- Calcolo dei Centroidi (Vettori Medi) ---")
# il controllo è uno solo, centroide vs centroide, il vettore media di tutte le righe
def calculate_all_centroids(X_data, y_target):

    # Crea un DataFrame temporaneo per sfruttare la velocità di Pandas GroupBy
    df_temp = pd.DataFrame(X_data)
    df_temp['Target'] = y_target.to_numpy()  # Usa to_numpy() per stabilità

    # Raggruppa per target (colonna 'Attack') e calcola la media per ogni feature
    centroids_df = df_temp.groupby('Target').mean()

    # Estrai gli indici delle classi che sono stati trovati
    unique_classes_found = centroids_df.index.to_numpy()

    # Restituisce l'array NumPy dei centroidi e i loro indici numerici
    return centroids_df.to_numpy(), unique_classes_found

# Esecuzione del calcolo
centroids, unique_classes_indices = calculate_all_centroids(X_normalized, y_multi) #usa tutte le colonne e il target

# Calcolo della Similarità del Coseno
similarity_matrix = cosine_similarity(centroids)

print(f"Centroidi calcolati per {centroids.shape[0]} classi.")
print(f"Tempo calcolo Centroidi: {time.time() - start_time:.2f} secondi")
e togli ciò che c'è dopo
"""
df_full = pd.DataFrame(X_normalized)
df_full[TARGET_COL_MULTI] = y_multi.to_numpy()
# Preparazione dei campioni per tutte le classi
sampled_vectors = {}
unique_classes = []

print(f"\n--- 1. Campionamento delle Classi (N={N_SAMPLES}) ---")

# Cicla su tutti i nomi di classe per campionare i vettori
for attack_name in DISPLAY_LABELS_MULTI:

    # Ottieni l'indice numerico della classe
    class_idx = np.where(DISPLAY_LABELS_MULTI == attack_name)[0][0]

    # Filtra il DataFrame per l'indice numerico
    df_class = df_full[df_full[TARGET_COL_MULTI] == class_idx]
    n_total = len(df_class)

    if n_total == 0:
        print(f"Classe {attack_name} saltata (0 campioni).")
        continue

    # Determina il numero di campioni da prendere
    n_sample_current = min(N_SAMPLES, n_total)

    # Campionamento
    X_sample = df_class.sample(n=n_sample_current, random_state=42).drop(TARGET_COL_MULTI, axis=1).to_numpy()

    sampled_vectors[attack_name] = X_sample
    unique_classes.append(attack_name)
    print(f"Campionato {attack_name} con {n_sample_current} vettori.")

# Inizializzazione della matrice di risultati
num_unique_classes = len(unique_classes)
similarity_matrix_mean = np.zeros((num_unique_classes, num_unique_classes))

print("\n--- Calcolo Matrice Vettore-Vettore ---")

# Ciclo incrociato su tutte le coppie di classi
for i in range(num_unique_classes):
    for j in range(num_unique_classes):

        class_A = unique_classes[i]
        class_B = unique_classes[j]

        X_A = sampled_vectors[class_A]
        X_B = sampled_vectors[class_B]

        # Calcolo della similarità vettore-vettore
        # Risultato: matrice (N_A x N_B) di tutti i confronti
        matrix = cosine_similarity(X_A, X_B)

        # Estrai la media di tutti i confronti
        mean_similarity = np.mean(matrix)
        similarity_matrix_mean[i, j] = mean_similarity


print(f"Tempo calcolo Matrice Vettore-Vettore: {time.time() - start_time:.2f} secondi")
print("\n--- Matrice di Similarità Completa ---")

#num_unique_classes = centroids.shape[0] decommenta per centroidi
class_indices = np.arange(num_unique_classes)

# Estrai i nomi delle classi basandoti sugli indici trovati
#class_names = [DISPLAY_LABELS_MULTI[i] for i in unique_classes_indices] same

fig, ax = plt.subplots(figsize=(14, 12))

im = ax.imshow(similarity_matrix_mean, cmap='Purples', interpolation='nearest', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Similarità del Coseno (1.0 = Somiglianza Massima)')

# Configura gli assi
ax.set_xticks(class_indices)
ax.set_yticks(class_indices)
ax.set_xticklabels(unique_classes, rotation=90) #cambia con class_names
ax.set_yticklabels(unique_classes) #same
ax.set_title("Matrice di Similarità tra tutte le Classi",
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Classe")
ax.set_ylabel("Classe")

# Stampa i valori numerici all'interno delle celle
threshold = 0.5
for i in range(num_unique_classes):
    for j in range(num_unique_classes):
        val = similarity_matrix_mean[i, j]
        color = "black" if val < threshold else "white"

        ax.text(j, i, f"{val:.3f}",
                ha="center", va="center", color=color, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nTempo totale di esecuzione: {time.time() - start_time:.2f} secondi")


