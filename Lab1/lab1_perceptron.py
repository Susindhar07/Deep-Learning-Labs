"""
Lab 1: Perceptron Learning Implementation
Course: Deep Learning (22AIE304) | Batch: 2026-2027
Description: Implements the Perceptron learning algorithm from scratch using
             NumPy for binary classification on the AND gate dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# 1. PERCEPTRON CORE COMPONENTS
# ─────────────────────────────────────────────

def activation(linear_output):
    """Heaviside Step Function: returns 1 if input >= 0, else 0."""
    return 1 if linear_output >= 0 else 0

def predict(x, weights, bias):
    """Compute linear combination and apply activation."""
    linear_output = np.dot(weights, x) + bias
    return activation(linear_output)

# ─────────────────────────────────────────────
# 2. PERCEPTRON TRAINING LOOP
# ─────────────────────────────────────────────

def train_perceptron(X, y, lr=0.1, epochs=10):
    """
    Train a Perceptron using the weight update rule.

    Parameters:
        X      : Input feature matrix (n_samples x n_features)
        y      : Target labels (binary: 0 or 1)
        lr     : Learning rate (η)
        epochs : Number of training iterations

    Returns:
        weights : Final weight vector
        bias    : Final bias
        history : List of (epoch, errors) tuples
    """
    weights = np.zeros(X.shape[1])   # w = [0, 0]
    bias = 0                          # b = 0
    history = []

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            # Step 1: Calculate Linear Combination  →  z = w·x + b
            linear_output = np.dot(weights, X[i]) + bias

            # Step 2: Apply Activation Function  →  ŷ = step(z)
            y_pred = 1 if linear_output >= 0 else 0

            # Step 3: Compute Update  →  Δw = η * (y - ŷ)
            update = lr * (y[i] - y_pred)

            # Step 4: Update Weights and Bias
            weights += update * X[i]   # w ← w + Δw · x
            bias    += update          # b ← b + Δw

            total_error += int(update != 0)

        history.append((epoch + 1, total_error))
        print(f"  Epoch {epoch+1:2d} | Errors: {total_error} | "
              f"Weights: {weights} | Bias: {bias:.1f}")

        # Early stopping if perfectly learned
        if total_error == 0:
            print(f"\n  ✓ Converged at epoch {epoch+1}!")
            break

    return weights, bias, history

# ─────────────────────────────────────────────
# 3. DATASET — AND GATE
# ─────────────────────────────────────────────

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])   # AND gate truth table

print("=" * 55)
print("  Lab 1 — Perceptron Learning (AND Gate)")
print("=" * 55)
print("\n  Dataset (AND Gate):")
print("  x1  x2  |  y")
print("  ─────────────")
for xi, yi in zip(X, y):
    print(f"   {xi[0]}   {xi[1]}  |  {yi}")

print("\n  Training Log:")
print("  " + "─" * 53)

weights, bias, history = train_perceptron(X, y, lr=0.1, epochs=10)

# ─────────────────────────────────────────────
# 4. OUTPUT ANALYSIS
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  OUTPUT ANALYSIS")
print("=" * 55)
print(f"  Dataset Used          : AND Gate")
print(f"  Final Weights (w1,w2) : {weights}")
print(f"  Final Bias (b)        : {bias}")
print(f"  Decision Boundary     : {weights[0]:.1f}·x1 + "
      f"{weights[1]:.1f}·x2 + ({bias:.1f}) = 0")

print("\n  Verification Table:")
print("  x1  x2  |  ŷ  |  y  | Correct?")
print("  ─────────────────────────────────")
correct = 0
for xi, yi in zip(X, y):
    y_hat = predict(xi, weights, bias)
    ok = "✓" if y_hat == yi else "✗"
    correct += int(y_hat == yi)
    print(f"   {xi[0]}   {xi[1]}  |  {y_hat}   |  {yi}  |   {ok}")

accuracy = correct / len(y) * 100
print(f"\n  Accuracy: {correct}/{len(y)} = {accuracy:.0f}%")

# ─────────────────────────────────────────────
# 5. DECISION BOUNDARY VISUALIZATION
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Lab 1 — Perceptron: AND Gate", fontsize=14, fontweight="bold")

# ── Plot A: Decision Boundary ──────────────────
ax = axes[0]
colors = ["#EF5350" if yi == 0 else "#42A5F5" for yi in y]
markers = ["o" if yi == 0 else "s" for yi in y]
for xi, yi, c, m in zip(X, y, colors, markers):
    ax.scatter(xi[0], xi[1], c=c, marker=m, s=120, zorder=5, edgecolors="k")
    ax.annotate(f"  ({xi[0]},{xi[1]})→{yi}", xy=(xi[0], xi[1]),
                fontsize=9, va="center")

# Decision boundary: w1*x1 + w2*x2 + b = 0  →  x2 = (-w1*x1 - b) / w2
x_line = np.linspace(-0.3, 1.3, 100)
if weights[1] != 0:
    y_line = (-weights[0] * x_line - bias) / weights[1]
    ax.plot(x_line, y_line, "k--", linewidth=2, label="Decision Boundary")

ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
ax.set_title("Decision Boundary")
ax.grid(True, alpha=0.3)
red_p  = mpatches.Patch(color="#EF5350", label="y=0 (AND=False)")
blue_p = mpatches.Patch(color="#42A5F5", label="y=1 (AND=True)")
ax.legend(handles=[red_p, blue_p, plt.Line2D([],[], color="k",
          linestyle="--", label="Decision Boundary")], fontsize=8)

# ── Plot B: Training Error Curve ───────────────
ax2 = axes[1]
epochs_list = [h[0] for h in history]
errors_list = [h[1] for h in history]
ax2.plot(epochs_list, errors_list, "o-", color="#26A69A", linewidth=2,
         markersize=7, markerfacecolor="white", markeredgewidth=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Number of Misclassifications")
ax2.set_title("Training Error vs Epoch")
ax2.set_xticks(epochs_list)
ax2.set_ylim(-0.2, max(errors_list) + 0.5)
ax2.grid(True, alpha=0.3)
ax2.fill_between(epochs_list, errors_list, alpha=0.15, color="#26A69A")

plt.tight_layout()
plt.savefig("lab1_perceptron_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Figure saved → lab1_perceptron_output.png")

# ─────────────────────────────────────────────
# 6. CRITICAL THINKING — XOR DISCUSSION
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  CRITICAL THINKING: XOR & Perceptron Convergence Theorem")
print("=" * 55)
print("""
  If XOR data is fed to this Perceptron:
  ────────────────────────────────────────────────────
  • XOR is NOT linearly separable — no single straight
    line can separate (0,1) and (1,0) from (0,0) and (1,1).

  • The Perceptron Convergence Theorem guarantees that the
    algorithm converges ONLY IF the data is linearly
    separable.

  • For XOR, the weights will oscillate indefinitely across
    every epoch and NEVER reach zero error.

  • Solution: Use a Multi-Layer Perceptron (MLP) with a
    hidden layer, which can learn non-linear decision
    boundaries via backpropagation (Lab 2).
""")
