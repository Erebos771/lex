import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
df = pd.read_csv("Wholesale customers data.csv")

print(df.head())

# -----------------------------
# 2. SELECT FEATURES
# -----------------------------
# Remove non-numeric columns if needed
X = df.drop(columns=['Channel', 'Region'])

# -----------------------------
# 3. SCALE DATA (VERY IMPORTANT)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. APPLY DBSCAN
# -----------------------------
db = DBSCAN(eps=1.5, min_samples=5)
labels = db.fit_predict(X_scaled)

# -----------------------------
# 5. ADD CLUSTER LABELS
# -----------------------------
df['Cluster'] = labels

print("\nCluster counts:")
print(df['Cluster'].value_counts())

# -----------------------------
# 6. VISUALIZATION (2 FEATURES)
# -----------------------------
plt.figure(figsize=(8,6))

for label in set(labels):
    if label == -1:
        color = 'black'
        name = 'Noise'
    else:
        color = plt.cm.tab10(label)
        name = f'Cluster {label}'

    plt.scatter(
        X_scaled[labels == label, 0],
        X_scaled[labels == label, 1],
        c=[color],
        label=name
    )

plt.legend()
plt.title("DBSCAN Clustering with Noise")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()