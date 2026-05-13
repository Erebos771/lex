import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

sns.set(style="whitegrid")

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------------
# 2. ELBOW METHOD
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# -----------------------------
# 3. APPLY K-MEANS (k=5)
# -----------------------------
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X)   # ✅ FIXED

# -----------------------------
# 4. CLUSTER SUMMARY (NOW WORKS)
# -----------------------------
df['Cluster'] = labels + 1

summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].agg(['mean', 'count'])
summary.columns = ['Avg Income', 'Count_Income', 'Avg Score', 'Count_Score']
summary = summary[['Avg Income', 'Avg Score', 'Count_Income']].rename(columns={'Count_Income': 'Size'})

print("\n── Cluster Summary ──────────────────────────")
print(summary.round(1).to_string())

# -----------------------------
# 5. BETTER VISUALIZATION
# -----------------------------
plt.figure(figsize=(8,6))

colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(5):
    plt.scatter(
        X.iloc[labels == i, 0],
        X.iloc[labels == i, 1],
        color=colors[i],
        label=f'Cluster {i+1}'
    )

# Centroids
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    c='black',
    marker='o',
    label='Centroids'
)

plt.title("K-Means Clustering (Improved)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()