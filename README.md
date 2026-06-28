# Deep Learning Lab Series — 22AIE304
**Batch: 2026–2027 | Team Number: 32**

A complete implementation of all four Deep Learning lab experiments, progressing from a basic Perceptron to a full Deep Neural Network for MNIST digit classification.

---

## Repository Structure

```
├── lab1_perceptron.py          # Lab 1 — Perceptron from scratch (NumPy)
├── lab2_mlp.py                 # Lab 2 — MLP with Keras, Sigmoid vs ReLU
├── lab3_hyperopt.py            # Lab 3 — Grid Search hyperparameter tuning
├── lab4_dnn_mnist.py           # Lab 4 — DNN for MNIST digit classification
├── lab1_perceptron_output.png  # Lab 1 decision boundary + error curve
├── lab2_mlp_output.png         # Lab 2 loss & accuracy curves
├── lab3_hyperopt_output.png    # Lab 3 trial comparison plots
├── lab4_dnn_mnist_output.png   # Lab 4 confusion matrix & curves
└── README.md
```

---

## Lab Summaries

### Lab 1 — Perceptron Learning (NumPy)
- Implements Perceptron from scratch with weight update rule `Δw = η(y - ŷ)x`
- Dataset: AND gate
- Converges in **4 epochs**, achieving **100% accuracy**
- Final weights: `[0.2, 0.1]`, bias: `-0.2`
- Decision boundary: `0.2·x₁ + 0.1·x₂ − 0.2 = 0`

### Lab 2 — MLP with Hyperparameter Tuning (Keras)
- Solves XOR gate (non-linearly separable) with 1 hidden layer (8 neurons)
- Compares Sigmoid vs ReLU at lr = 0.01 and 0.1 over 100 epochs

| Exp | Activation | LR   | Final Loss | Final Acc |
|-----|------------|------|------------|-----------|
| 1   | Sigmoid    | 0.01 | 0.6911     | 50.0%     |
| 2   | Sigmoid    | 0.1  | 0.0072     | 100.0%    |
| 3   | ReLU       | 0.01 | 0.3399     | 100.0%    |
| 4   | ReLU       | 0.1  | 0.3471     | 75.0%     |

**Key finding:** Sigmoid + lr=0.1 and ReLU + lr=0.01 both converge to 100%. Sigmoid + lr=0.01 fails due to vanishing gradients.

### Lab 3 — Advanced Hyperparameter Optimization (Grid Search on Iris)
- Dataset: Iris (4 features, 3 classes, 150 samples)
- MLP: 2 hidden layers + Dropout, optimized with Adam
- 5 hyperparameter trials varying LR, hidden units, batch size, dropout

| Trial | LR    | Units | Batch | Dropout | Val Acc |
|-------|-------|-------|-------|---------|---------|
| 1     | 0.01  | 16    | 16    | 0.0     | 96.7%   |
| 2     | 0.01  | 32    | 16    | 0.2     | 96.7%   |
| 3     | 0.001 | 32    | 32    | 0.3     | 90.0%   |
| 4     | 0.01  | 64    | 16    | 0.3     | 96.7%   |
| 5     | 0.001 | 64    | 32    | 0.1     | 96.7%   |

**Optimal:** Trial 1 — LR=0.01, 16 hidden units, batch=16, dropout=0.0

### Lab 4 — DNN for MNIST Digit Classification (Keras + TensorFlow)
- Architecture: `Flatten(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`
- Optimizer: Adam | Loss: Sparse Categorical Cross-Entropy
- Trained for 5 epochs, batch size 32

**Expected Results:**
- Test Accuracy: **~97.9%**
- Test Loss: **~0.070**
- Most confused pairs: 4↔9, 3↔8, 7↔1

---

## Requirements

```bash
pip install numpy matplotlib tensorflow scikit-learn seaborn
```

Python 3.8+ required. For Lab 4, internet access is needed to auto-download MNIST (~11 MB).

---

## How to Run

```bash
# Run each lab independently
python lab1_perceptron.py
python lab2_mlp.py
python lab3_hyperopt.py
python lab4_dnn_mnist.py
```

Each script prints a structured output log and saves a PNG figure automatically.

---

## Final Synthesis — Performance Summary

| Lab | Experiment       | Accuracy | Notes                    |
|-----|-----------------|----------|--------------------------|
| 1   | Perceptron (AND) | 100%    | Linearly separable only  |
| 2   | MLP (XOR)        | 100%    | ReLU + lr=0.01           |
| 3   | Optimized MLP    | 96.7%   | Iris dataset, 3-class    |
| 4   | DNN (MNIST)      | ~97.9%  | 10-class digit recognition|

---

## Key Takeaways

1. **Perceptron** solves linearly separable problems only (AND/OR, not XOR).
2. **MLP with ReLU** overcomes XOR via non-linear hidden layers.
3. **Hyperparameter tuning** (batch size, dropout, LR) critically impacts generalization.
4. **Deep networks** (Lab 4) hierarchically extract features unavailable to shallow models.
5. The **Universal Approximation Theorem** is empirically demonstrated across the lab series.
