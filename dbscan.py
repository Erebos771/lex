import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

# -----------------------------
# 1. GENERATE DATASET
# -----------------------------
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# -----------------------------
# 2. SCALE DATA (IMPORTANT)
# -----------------------------
X_scaled = StandardScaler().fit_transform(X)

# -----------------------------
# 3. APPLY DBSCAN
# -----------------------------
db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X_scaled)

# -----------------------------
# 4. COUNT CLUSTERS & NOISE
# -----------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("Clusters:", n_clusters)
print("Noise points:", n_noise)

# -----------------------------
# 5. VISUALIZATION
# -----------------------------
plt.figure(figsize=(7,5))

# Color map
unique_labels = set(labels)

for label in unique_labels:
    if label == -1:
        color = 'black'
        label_name = 'Noise'
    else:
        color = plt.cm.tab10(label)
        label_name = f'Cluster {label}'

    plt.scatter(
        X_scaled[labels == label, 0],
        X_scaled[labels == label, 1],
        c=[color],
        label=label_name
    )

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()