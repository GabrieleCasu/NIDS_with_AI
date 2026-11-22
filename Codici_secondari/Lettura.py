import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

file_path = "../Dataset_secondari/NF-ToN-IoT-v3 - Copia.xlsx"

# Leggi il file Excel (una colonna unica)
df_raw = pd.read_excel(file_path, header=None, skiprows=1)

# Splitta la prima colonna in più colonne
df = df_raw.iloc[:,0].str.split(",", expand=True)

# Assegna i nomi delle colonne
columns = ["FLOW_START_MILLISECONDS","FLOW_END_MILLISECONDS","IPV4_SRC_ADDR","L4_SRC_PORT",
           "IPV4_DST_ADDR","L4_DST_PORT","PROTOCOL","L7_PROTO","IN_BYTES","IN_PKTS","OUT_BYTES",
           "OUT_PKTS","TCP_FLAGS","CLIENT_TCP_FLAGS","SERVER_TCP_FLAGS","FLOW_DURATION_MILLISECONDS",
           "DURATION_IN","DURATION_OUT","MIN_TTL","MAX_TTL","LONGEST_FLOW_PKT","SHORTEST_FLOW_PKT",
           "MIN_IP_PKT_LEN","MAX_IP_PKT_LEN","SRC_TO_DST_SECOND_BYTES","DST_TO_SRC_SECOND_BYTES",
           "RETRANSMITTED_IN_BYTES","RETRANSMITTED_IN_PKTS","RETRANSMITTED_OUT_BYTES","RETRANSMITTED_OUT_PKTS",
           "SRC_TO_DST_AVG_THROUGHPUT","DST_TO_SRC_AVG_THROUGHPUT","NUM_PKTS_UP_TO_128_BYTES",
           "NUM_PKTS_128_TO_256_BYTES","NUM_PKTS_256_TO_512_BYTES","NUM_PKTS_512_TO_1024_BYTES",
           "NUM_PKTS_1024_TO_1514_BYTES","TCP_WIN_MAX_IN","TCP_WIN_MAX_OUT","ICMP_TYPE","ICMP_IPV4_TYPE",
           "DNS_QUERY_ID","DNS_QUERY_TYPE","DNS_TTL_ANSWER","FTP_COMMAND_RET_CODE","SRC_TO_DST_IAT_MIN",
           "SRC_TO_DST_IAT_MAX","SRC_TO_DST_IAT_AVG","SRC_TO_DST_IAT_STDDEV","DST_TO_SRC_IAT_MIN",
           "DST_TO_SRC_IAT_MAX","DST_TO_SRC_IAT_AVG","DST_TO_SRC_IAT_STDDEV","Label","Attack"]

df.columns = columns

print(f" File letto con successo")
print(f" Dimensioni: {df.shape[0]} righe, {df.shape[1]} colonne")
print(f"\n Numero di attacchi: {df['Attack'].nunique()}")

# Encoding delle colonne stringa
le = LabelEncoder()
for col in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"]:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Features e target
X = df.drop(["Label", "Attack"], axis=1, errors="ignore")
y = df["Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest
rfc = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42) #più alberi=più lungo, più profondi=più lingo
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print(f" Accuracy: {accuracy_score(y_test, y_pred):.4f}")

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
disp.plot(cmap="Blues", values_format="d") #più leggibile e con valori interi
plt.title("Confusion matrix with only attack or benign")
plt.show()

#ora differenzio gli attacchi
attack_labels = le.classes_
y_multi = df["Attack"]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_multi, test_size=0.3, random_state=42)

rfc_multi = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rfc_multi.fit(X_train_m, y_train_m)
y_pred_multi = rfc_multi.predict(X_test_m)

print(f" Accuracy (multi-classe): {accuracy_score(y_test_m, y_pred_multi):.4f}")
cm_multi = confusion_matrix(y_test_m, y_pred_multi)
disp_multi = ConfusionMatrixDisplay(confusion_matrix=cm_multi, display_labels=attack_labels)
disp_multi.plot(cmap="Purples", values_format="d")
plt.title("Confusion Matrix with all the Attacks")
plt.xticks(rotation=90)
plt.show()
