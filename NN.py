import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

torch.manual_seed(42)
sns.set(style="whitegrid")

# ── DATA ───────────────────────────────────────────────────
X, y   = load_iris(return_X_y=True)
names  = load_iris().target_names
X      = StandardScaler().fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

to_t  = lambda a, dtype: torch.tensor(a, dtype=dtype)
X_tr  = to_t(X_tr, torch.float32);  X_te = to_t(X_te, torch.float32)
y_tr  = to_t(y_tr, torch.long);     y_te = to_t(y_te, torch.long)

# ── MODEL ──────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(16, 8), nn.BatchNorm1d(8),  nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(8, 3)
)

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# ── TRAIN ──────────────────────────────────────────────────
EPOCHS      = 150
train_losses, train_accs = [], []

for epoch in range(EPOCHS):
    model.train()

    logits = model(X_tr)
    loss   = loss_fn(logits, y_tr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # track metrics
    with torch.no_grad():
        preds     = logits.argmax(1)
        train_acc = (preds == y_tr).float().mean().item() * 100

    train_losses.append(loss.item())
    train_accs.append(train_acc)

    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.1f}%")

# ── EVALUATE ───────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred = model(X_te).argmax(1)

y_pred_np = y_pred.numpy()
y_te_np   = y_te.numpy()

print(f"\nTest Accuracy : {accuracy_score(y_te_np, y_pred_np)*100:.1f}%")
print("\n── Classification Report ──")
print(classification_report(y_te_np, y_pred_np, target_names=names))

# ── PLOTS ──────────────────────────────────────────────────
# 1. LOSS CURVE
plt.figure(figsize=(5,4))
plt.plot(train_losses, color="steelblue", linewidth=1.8)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 2. ACCURACY
test_acc = accuracy_score(y_te_np, y_pred_np) * 100

plt.figure(figsize=(5,4))
plt.plot(train_accs, color="seagreen", linewidth=1.8)
plt.axhline(test_acc, color="tomato", linestyle="--",
            label=f"Test: {test_acc:.1f}%")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# 3. CONFUSION MATRIX
cm = confusion_matrix(y_te_np, y_pred_np)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=names, yticklabels=names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()