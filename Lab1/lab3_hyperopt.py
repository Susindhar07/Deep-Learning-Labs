"""
Lab 3: Advanced Hyperparameter Optimization
Course: Deep Learning (22AIE304) | Batch: 2026-2027
Description: Grid Search over hyperparameters (learning rate, hidden units,
             batch size, dropout) on the Iris dataset using Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools, warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ─────────────────────────────────────────────
# PART 1: THEORETICAL ANALYSIS
# ─────────────────────────────────────────────

print("=" * 65)
print("  Lab 3 — Advanced Hyperparameter Optimization")
print("=" * 65)
print("""
  THEORETICAL ANALYSIS
  ════════════════════

  1. Impact of Batch Size:
  ─────────────────────────────────────────────────────────────
  • Small batch (e.g., 16): Noisy gradient estimates introduce
    stochasticity. This acts as implicit regularization, often
    improving generalization. Convergence is slower per epoch
    but can escape shallow local minima.
  • Large batch (e.g., 256): Smooth, accurate gradients lead to
    faster convergence per epoch but may converge to sharp
    minima that generalize poorly (the "generalization gap").
  • Rule of thumb: Smaller batches tend to generalize better;
    larger batches train faster on GPU.

  2. Role of Dropout Rate:
  ─────────────────────────────────────────────────────────────
  • Dropout randomly zeros neurons with probability p during
    training, forcing the network to learn redundant features.
  • At p=0.0: no regularization → model memorizes training data
    → overfitting.
  • At p=0.5: strong regularization → capacity reduced → less
    overfitting, but potential underfitting if model too small.
  • Optimal p is task-dependent; typically 0.2–0.5 for hidden
    layers.
""")

# ─────────────────────────────────────────────
# DATASET — IRIS (multi-class, 3 classes)
# ─────────────────────────────────────────────

iris = load_iris()
X_raw, y_raw = iris.data, iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

print(f"  Dataset: Iris | Features: 4 | Classes: 3")
print(f"  Train samples: {len(X_train)} | Val samples: {len(X_val)}\n")

# ─────────────────────────────────────────────
# PART 2: GRID SEARCH
# ─────────────────────────────────────────────

param_grid = {
    "learning_rate": [0.01, 0.001],
    "hidden_units":  [16, 32],
    "batch_size":    [16, 32],
    "dropout_rate":  [0.0, 0.3],
}

# Generate 5 representative combos (selected from grid)
selected_combos = [
    {"learning_rate": 0.01,  "hidden_units": 16, "batch_size": 16, "dropout_rate": 0.0},
    {"learning_rate": 0.01,  "hidden_units": 32, "batch_size": 16, "dropout_rate": 0.2},
    {"learning_rate": 0.001, "hidden_units": 32, "batch_size": 32, "dropout_rate": 0.3},
    {"learning_rate": 0.01,  "hidden_units": 64, "batch_size": 16, "dropout_rate": 0.3},
    {"learning_rate": 0.001, "hidden_units": 64, "batch_size": 32, "dropout_rate": 0.1},
]

EPOCHS = 80

def build_model(hidden_units, dropout_rate, learning_rate):
    model = models.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden_units // 2, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

print("=" * 65)
print("  OPTIMIZATION LOG (Grid Search — 5 Trials)")
print("=" * 65)
print(f"  {'Trial':<6} {'LR':<7} {'Units':<7} {'Batch':<7} {'Drop':<7} "
      f"{'TrainLoss':<11} {'ValAcc':<9} {'Overfit?'}")
print("  " + "─" * 63)

trial_results = []
best_val_acc = 0
best_config = None

for i, params in enumerate(selected_combos, 1):
    tf.random.set_seed(42); np.random.seed(42)
    m = build_model(params["hidden_units"],
                    params["dropout_rate"],
                    params["learning_rate"])
    hist = m.fit(X_train, y_train,
                 epochs=EPOCHS,
                 batch_size=params["batch_size"],
                 validation_data=(X_val, y_val),
                 verbose=0)

    train_loss = hist.history["loss"][-1]
    train_acc  = hist.history["accuracy"][-1]
    val_acc    = hist.history["val_accuracy"][-1]
    gap        = train_acc - val_acc
    overfit    = "Yes" if gap > 0.10 else "No"

    trial_results.append({
        **params,
        "trial": i,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "overfit": overfit,
        "history": hist.history
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_config = {**params, "trial": i}

    print(f"  {i:<6} {params['learning_rate']:<7} {params['hidden_units']:<7} "
          f"{params['batch_size']:<7} {params['dropout_rate']:<7.1f} "
          f"{train_loss:<11.4f} {val_acc*100:<9.1f} {overfit}")

# ─────────────────────────────────────────────
# OPTIMAL CONFIGURATION SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print(f"  OPTIMAL CONFIGURATION — Trial {best_config['trial']}")
print("=" * 65)
print(f"  Learning Rate : {best_config['learning_rate']}")
print(f"  Hidden Units  : {best_config['hidden_units']}")
print(f"  Batch Size    : {best_config['batch_size']}")
print(f"  Dropout Rate  : {best_config['dropout_rate']}")
print(f"  Val Accuracy  : {best_val_acc*100:.1f}%")
print("""
  Justification:
  ─────────────────────────────────────────────────────────
  • The best configuration balances model capacity (hidden
    units) with regularization (dropout) and a learning rate
    that converges stably without oscillation.
  • Dropout prevents overfitting by reducing co-adaptation
    between neurons, while a moderate batch size maintains
    stochastic noise for better generalization.
""")

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Lab 3 — Hyperparameter Optimization on Iris Dataset",
             fontsize=13, fontweight="bold")

palette = ["#EF5350","#FF8A65","#42A5F5","#26A69A","#AB47BC"]

ax = axes[0]
for r in trial_results:
    ax.plot(r["history"]["loss"], color=palette[r["trial"]-1],
            linewidth=2, label=f"T{r['trial']}: lr={r['learning_rate']}, "
            f"d={r['dropout_rate']}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
ax.set_title("Training Loss — All Trials")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax2 = axes[1]
for r in trial_results:
    ax2.plot([v*100 for v in r["history"]["val_accuracy"]],
             color=palette[r["trial"]-1], linewidth=2,
             label=f"T{r['trial']}: units={r['hidden_units']}, "
             f"bs={r['batch_size']}")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Accuracy (%)")
ax2.set_title("Validation Accuracy — All Trials")
ax2.set_ylim(0, 105)
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
# Mark best trial
best_hist = trial_results[best_config["trial"]-1]["history"]["val_accuracy"]
ax2.axhline(y=max(best_hist)*100, color="gold", linewidth=1.5,
            linestyle="--", label=f"Best Val Acc: {max(best_hist)*100:.1f}%")
ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig("lab3_hyperopt_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Figure saved → lab3_hyperopt_output.png")
