import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys  # To exit in case of error

# Path to the CORRECT and COMPLETE CSV file (used only the first time)
CSV_FILE_PATH = "../Dataset_secondari/NF-ToN-IoT-v3_CORRECTED_FAST.csv"
# Path to save/load the fast binary file (Parquet)
PARQUET_FILE_PATH = "../Dataset/NF-ToN-IoT-v3_Complete.parquet"

TARGET_COL_BINARIO = "Label"
TARGET_COL_MULTI = "Attack"
CATEGORICAL_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]

# STATIC ARRAYS OF AXIS NAMES
DISPLAY_LABELS_MULTI = np.array(['Backdoor', 'Benign', 'ddos', 'dos', 'injection', 'mitm', 'password',
 'ransomware', 'scanning', 'xss'])
DISPLAY_LABELS_BINARIO = np.array(['Benign', 'Attack'])

# Names of the 55 columns
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
    # SCENARIO 1: FAST LOADING from Parquet
    print(f"Parquet file found. Fast loading from: {PARQUET_FILE_PATH}")
    try:
        df = pd.read_parquet(PARQUET_FILE_PATH)
        # Removed: initialize_label_encoders(df)
        load_end_time = time.time()
        print(f"Loading from Parquet completed. Time: {load_end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"ERROR loading Parquet ({e}). Falling back to CSV loading.")
        df = None

if df is None:
    # SCENARIO 2: SLOW LOADING - Exit if Parquet not found
    print("ERROR: Could not load DataFrame from CSV or Parquet.")
    sys.exit(1)

# Final Checks
print(f"\nDataset Final Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Total Loading/Pre-processing Time: {time.time() - start_time:.2f} seconds")


 # BINARY CLASSIFICATION (Target: Label)

print("\n--- 3. Binary Classification (Label) ---")

X = df.drop([TARGET_COL_BINARIO, TARGET_COL_MULTI], axis=1, errors="ignore")
y = df[TARGET_COL_BINARIO]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MEMORY OPTIMIZATION: Convert to 32-bit (Crucial for stability)
X_train_32 = X_train.astype(np.float32)
y_train_32 = y_train.astype(np.int32)

# Optimized Random Forest (n_jobs=7, perhaps more feasible, 10 seems stable)
rfc = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=10)

train_start_time = time.time()
print(f"Starting binary training on {X_train.shape[0]} rows...")
rfc.fit(X_train_32, y_train_32)
train_end_time = time.time()

y_pred = rfc.predict(X_test.astype(np.float32))

print(f"Training Time (Binary): {train_end_time - train_start_time:.2f} seconds")
print(f"Accuracy (Binary): {accuracy_score(y_test.astype(np.int32), y_pred):.4f}")

# Binary confusion matrix
cm = confusion_matrix(y_test.astype(np.int32), y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS_BINARIO)
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Binary)")
plt.show()

# MULTI-CLASS CLASSIFICATION (Target: Attack)

print("\n--- 4. Multi-Class Classification (Attack) ---")
y_multi = df[TARGET_COL_MULTI]

# Train/test split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_multi, test_size=0.3, random_state=42)

# MEMORY OPTIMIZATION: Convert to 32-bit (Crucial)
X_train_m_32 = X_train_m.astype(np.float32)
y_train_m_32 = y_train_m.astype(np.int32)

# Random Forest (n_jobs=1 for maximum stability, probably can be increased, 2 seems okay)
rfc_multi = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=2)

train_start_time_m = time.time()
print(f"Starting multi-class training on {X_train_m.shape[0]} rows ...")
rfc_multi.fit(X_train_m_32, y_train_m_32)
train_end_time_m = time.time()

y_pred_multi = rfc_multi.predict(X_test_m.astype(np.float32))

print(f"Training Time (Multi-class): {train_end_time_m - train_start_time_m:.2f} seconds")
print(f"Accuracy (Multi-class): {accuracy_score(y_test_m.astype(np.int32), y_pred_multi):.4f}")

# Multi-class confusion matrix
cm_multi = confusion_matrix(y_test_m.astype(np.int32), y_pred_multi)

# Initialize Display with REAL attack NAMES (Static Array)
disp_multi = ConfusionMatrixDisplay(confusion_matrix=cm_multi, display_labels=DISPLAY_LABELS_MULTI)

# Plot Configuration
fig, ax = plt.subplots(figsize=(16, 12))

# PLOT: Exponential format
disp_multi.plot(cmap="Purples", values_format=".2e", ax=ax)
ax.set_title("Confusion Matrix - FULL Dataset")

# Forced Reduction of Cell Font, otherwise values overflow
for row in disp_multi.text_:
    for text in row:
        text.set_fontsize(7) # Font 7 to fit the exponential numbers

plt.xticks(rotation=90) # Rotate X names
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
print("\n--- Analysis Completed ---")