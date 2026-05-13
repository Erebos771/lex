import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate two moons data
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train SVM with polynomial kernel
svm = SVC(kernel='poly', degree=5, C=1.0, coef0=1)
svm.fit(X_train, y_train)

# 4. Accuracy
y_pred = svm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
print(f"Support vectors: {svm.n_support_}")

# 5. Plot decision boundary
def plot_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.contour(xx, yy, Z, colors='gray', linewidths=1)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='steelblue', label='Class 0', edgecolors='k', s=40)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='tomato',    label='Class 1', edgecolors='k', s=40)

    # Highlight support vectors
    sv = model.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=120, facecolors='none', edgecolors='black', linewidths=1.5, label='Support vectors')

    plt.title(f"SVM — Polynomial Kernel (degree=3)  |  Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_boundary(svm, X, y)