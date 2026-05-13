import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data.csv")

# -----------------------------
# 2. SELECT FEATURES
# -----------------------------
# For visualization (2 features only)
X = df[['Glucose', 'BMI']]
y = df['Outcome']

# -----------------------------
# 3. TRAIN SVM CLASSIFICATION
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("=== CLASSIFICATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 4. DECISION BOUNDARY
# -----------------------------
xx, yy = np.meshgrid(
    np.linspace(X.iloc[:,0].min(), X.iloc[:,0].max(), 100),
    np.linspace(X.iloc[:,1].min(), X.iloc[:,1].max(), 100)
)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y)

plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("SVM Decision Boundary")
plt.show()

# -----------------------------
# 5. SVM REGRESSION
# -----------------------------
X_reg = df[['Glucose']]
y_reg = df['BMI']

svr = SVR(kernel='linear')
svr.fit(X_reg, y_reg)

y_pred_reg = svr.predict(X_reg)

print("\n=== REGRESSION ===")
print("MSE:", mean_squared_error(y_reg, y_pred_reg))

plt.scatter(X_reg, y_reg)
plt.plot(X_reg, y_pred_reg)
plt.title("SVM Regression")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.show()