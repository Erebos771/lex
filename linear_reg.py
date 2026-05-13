import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

X = df[['rm']].values   # feature
y = df['medv'].values   # target

# -----------------------------
# 2. Z-score normalization (manual)
# -----------------------------
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean) / X_std

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. TRAIN Linear Regression (from scratch)
# y = mx + c
# -----------------------------
m = 0
c = 0
lr = 0.01
epochs = 1000
n = len(X_train)

for _ in range(epochs):
    y_pred_train = m * X_train.flatten() + c
    
    # gradients
    dm = (-2/n) * np.sum(X_train.flatten() * (y_train - y_pred_train))
    dc = (-2/n) * np.sum(y_train - y_pred_train)
    
    # update
    m = m - lr * dm
    c = c - lr * dc

print("Trained parameters:")
print("m =", m, "c =", c)

# -----------------------------
# 5. Prediction
# -----------------------------
y_pred = m * X_test.flatten() + c

# -----------------------------
# 6. Regression Metrics
# -----------------------------
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# -----------------------------
# 7. Regression Graph
# -----------------------------
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.title("Linear Regression (Trained from Scratch)")
plt.xlabel("Rooms (Z-score)")
plt.ylabel("Price")
plt.show()

# -----------------------------
# 8. Convert to Classification (for CM etc.)
# -----------------------------
threshold = np.mean(y)

y_test_class = (y_test > threshold).astype(int)
y_pred_class = (y_pred > threshold).astype(int)

cm = confusion_matrix(y_test_class, y_pred_class)

print("Confusion Matrix:\n", cm)
print("Precision:", precision_score(y_test_class, y_pred_class))
print("Recall:", recall_score(y_test_class, y_pred_class))
print("F1 Score:", f1_score(y_test_class, y_pred_class))

# -----------------------------
# 9. Confusion Matrix Graph
# -----------------------------
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()