import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

# -----------------------------
# 1. Load Dataset (LOCAL FILE)
# -----------------------------
df = pd.read_csv("Titanic-Dataset.csv")

# -----------------------------
# 2. Explore Data
# -----------------------------
print(df.head())
print(df.info())
print(df.isnull().sum())

# Select useful columns
df = df[['Survived', 'Pclass', 'Age', 'Fare']].dropna()

X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# -----------------------------
# 6. Evaluation Metrics
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))





X_vis = df[['Age', 'Fare']]
y_vis = df['Survived']

# Train model again on 2 features
model_vis = LogisticRegression()
model_vis.fit(X_vis, y_vis)

# Create mesh grid
x_min, x_max = X_vis['Age'].min() - 1, X_vis['Age'].max() + 1
y_min, y_max = X_vis['Fare'].min() - 1, X_vis['Fare'].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

# Predict over grid
Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot actual data points
plt.scatter(X_vis['Age'], X_vis['Fare'], c=y_vis)

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Decision Boundary (Separation Graph)")
plt.show()

# -----------------------------
# 7. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 8. ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()