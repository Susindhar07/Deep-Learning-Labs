"""
Lab 4: Deep Neural Network (DNN) for MNIST Digit Classification
Course: Deep Learning (22AIE304) | Batch: 2026-2027
Description: Trains a DNN on MNIST, evaluates with confusion matrix,
             performs error analysis, and produces all required visualizations.

NOTE: Run this script in an environment with internet access so that
      tf.keras.datasets.mnist.load_data() can download the dataset (~11 MB).
      The code below is complete and ready to run as-is.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

print("=" * 60)
print("  Lab 4 — DNN for MNIST Digit Classification")
print("=" * 60)

# ─────────────────────────────────────────────
# PART 1: LOAD AND PREPROCESS DATA
# ─────────────────────────────────────────────

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0–255) → (0.0–1.0)
x_train = x_train / 255.0
x_test  = x_test  / 255.0

print(f"\n  Data loaded and normalized.")
print(f"  x_train shape : {x_train.shape}  (60,000 images, 28×28)")
print(f"  x_test  shape : {x_test.shape}   (10,000 images, 28×28)")
print(f"  Pixel range   : [{x_train.min():.1f}, {x_train.max():.1f}]")

# ─────────────────────────────────────────────
# PART 2: BUILD DNN ARCHITECTURE
# ─────────────────────────────────────────────

model = models.Sequential([
    # Flatten 28×28 image → 784-dim vector
    layers.Flatten(input_shape=(28, 28)),

    # Hidden Layer 1: 128 neurons + ReLU
    layers.Dense(128, activation="relu"),

    # Hidden Layer 2: 64 neurons + ReLU
    layers.Dense(64, activation="relu"),

    # Output Layer: 10 classes (digits 0–9) + Softmax
    layers.Dense(10, activation="softmax")
])

# ─────────────────────────────────────────────
# PART 3: COMPILE THE MODEL
# ─────────────────────────────────────────────

model.compile(
    optimizer="adam",                        # Adaptive Moment Estimation
    loss="sparse_categorical_crossentropy",  # integer labels
    metrics=["accuracy"]
)

print("\n  Model Architecture:")
model.summary()

# ─────────────────────────────────────────────
# PART 4: TRAIN THE MODEL
# ─────────────────────────────────────────────

print("\n  Training (5 epochs)...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ─────────────────────────────────────────────
# PART 5: EVALUATE ON TEST SET
# ─────────────────────────────────────────────

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
y_pred_probs = model.predict(x_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

print(f"\n  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc*100:.2f}%")

# ─────────────────────────────────────────────
# PART 6: CONFUSION MATRIX
# ─────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  CONFUSION MATRIX ANALYSIS")
print("=" * 60)
print(f"\n  Primary diagonal (correct predictions per digit):")
for i in range(10):
    print(f"    Digit {i}: {cm[i,i]:5d} / {cm[i].sum():5d}  "
          f"({cm[i,i]/cm[i].sum()*100:.1f}%)")

off_diag = [(i, j, cm[i,j]) for i in range(10)
            for j in range(10) if i != j]
off_diag.sort(key=lambda x: x[2], reverse=True)
print("\n  Top-5 most confused digit pairs:")
print(f"  {'True':>6} → {'Predicted':>9} | {'Count':>6}")
print("  " + "─" * 32)
for (true, pred, cnt) in off_diag[:5]:
    print(f"    {true:>4}  → {pred:>9}  |  {cnt:>5}")

# ─────────────────────────────────────────────
# PART 7: ERROR ANALYSIS — 3 MISCLASSIFIED IMAGES
# ─────────────────────────────────────────────

misclassified  = np.where(y_pred != y_test)[0]
sample_errors  = misclassified[:3]

print("\n" + "=" * 60)
print("  ERROR ANALYSIS — 3 Misclassified Images")
print("=" * 60)
print(f"\n  Total misclassified: {len(misclassified)} / {len(y_test)}")
print(f"\n  {'Image ID':<10} {'True Label':<13} {'Predicted':<12} {'Confidence'}")
print("  " + "─" * 50)
for idx in sample_errors:
    true_l = y_test[idx]
    pred_l = y_pred[idx]
    conf   = y_pred_probs[idx][pred_l] * 100
    print(f"  {idx:<10} {true_l:<13} {pred_l:<12} {conf:.1f}%")

print("""
  Visual Characteristics (Why did the DNN fail?):
  ─────────────────────────────────────────────────
  • Digits like 4 and 9 share similar upper loops.
  • Digits like 3 and 8 share curved upper/lower arcs.
  • Digits like 1 and 7 differ mainly in a horizontal bar.
  • Poorly written / rotated / thin-stroked digits fool
    the fully-connected layers (no spatial invariance).
""")

# ─────────────────────────────────────────────
# PART 8: VISUALIZATIONS
# ─────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Lab 4 — DNN for MNIST Digit Classification",
             fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# A: Loss Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(range(1,6), history.history["loss"],      "o-", color="#EF5350",
         linewidth=2, label="Train Loss")
ax1.plot(range(1,6), history.history["val_loss"],  "s--", color="#FF8A65",
         linewidth=2, label="Val Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Loss Curve"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# B: Accuracy Curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(range(1,6), [a*100 for a in history.history["accuracy"]],
         "o-", color="#42A5F5", linewidth=2, label="Train Acc")
ax2.plot(range(1,6), [a*100 for a in history.history["val_accuracy"]],
         "s--", color="#26A69A", linewidth=2, label="Val Acc")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy Curve"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# C: Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.3, linecolor="gray", cbar=False,
            annot_kws={"size": 6})
ax3.set_xlabel("Predicted"); ax3.set_ylabel("True")
ax3.set_title("Confusion Matrix")

# D: 3 Misclassified samples
for k, idx in enumerate(sample_errors):
    ax = fig.add_subplot(gs[1, k])
    ax.imshow(x_test[idx], cmap="gray")
    ax.set_title(f"ID:{idx}\nTrue:{y_test[idx]} → Pred:{y_pred[idx]}\n"
                 f"Conf:{y_pred_probs[idx][y_pred[idx]]*100:.1f}%", fontsize=8)
    ax.axis("off")

# E: Digit gallery
for j in range(3):
    ax = fig.add_subplot(gs[2, j])
    d1, d2 = j*2, j*2+1
    idx1 = np.where(y_test == d1)[0][0]
    idx2 = np.where(y_test == d2)[0][0]
    combined = np.concatenate([x_test[idx1], x_test[idx2]], axis=1)
    ax.imshow(combined, cmap="gray")
    ax.set_title(f"Digit {d1}  |  Digit {d2}", fontsize=8)
    ax.axis("off")

plt.savefig("lab4_dnn_mnist_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Figure saved → lab4_dnn_mnist_output.png")

# Classification Report
print("\n" + "=" * 60)
print("  CLASSIFICATION REPORT (Per Digit)")
print("=" * 60)
print(classification_report(y_test, y_pred,
                             target_names=[f"Digit {i}" for i in range(10)]))

# Critical Reflection
print("=" * 60)
print("  CRITICAL REFLECTION")
print("=" * 60)
print("""
  Depth & Activation vs. Single-Layer Perceptron (Lab 1):
  ─────────────────────────────────────────────────────────
  • Lab 1 Perceptron: single linear layer → ONLY linearly
    separable problems solvable. Cannot classify MNIST.

  • Lab 4 DNN (Flatten → 128 → 64 → 10, all ReLU except output):
    - Layer 1 (128): learns low-level features (edges, curves).
    - Layer 2 (64):  learns mid-level combinations (loops, bars).
    - Layer 3 (10):  maps to digit class probabilities.
    - ReLU avoids vanishing gradients, enabling deep training.
    - Softmax output gives calibrated class probabilities.

  • Result: ~97–98% test accuracy vs. Perceptron's inability
    to classify MNIST — demonstrating the power of depth
    and non-linear activations for hierarchical learning.
""")
