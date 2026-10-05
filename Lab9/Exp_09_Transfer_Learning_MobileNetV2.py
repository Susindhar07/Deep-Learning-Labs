# Experiment 9: Transfer Learning using Pretrained MobileNetV2
# File: Exp_09_Transfer_Learning_MobileNetV2.ipynb
# Student Roll Number: CH.SC.U4AIE24002

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("=" * 60)
print("Student Roll Number: CH.SC.U4AIE24002")
print("Experiment 9: Transfer Learning & Fine-Tuning with MobileNetV2")
print("=" * 60)

(X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.cifar10.load_data()

X_train = X_train_full[:5000]
y_train = y_train_full[:5000]
X_test = X_test_full[:1000]
y_test = y_test_full[:1000]

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Resizing(96, 96)(inputs)
x = layers.Lambda(preprocess_input)(x)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs)

print(f"\n[Roll No: CH.SC.U4AIE24002] Phase 1: Training Classification Head (Base Frozen)...")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_phase1 = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)

loss_p1, acc_p1 = model.evaluate(X_test, y_test, verbose=0)
print(f"[Roll No: CH.SC.U4AIE24002] Phase 1 Test Accuracy: {acc_p1 * 100:.2f}%")

print(f"\n[Roll No: CH.SC.U4AIE24002] Phase 2: Unfreezing Last 20 Layers for Fine-Tuning...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_phase2 = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test), verbose=1)

loss_p2, acc_p2 = model.evaluate(X_test, y_test, verbose=0)
print("\n" + "=" * 60)
print(f"Transfer Learning Results [Roll No: CH.SC.U4AIE24002]:")
print(f"Initial Frozen Accuracy:    {acc_p1 * 100:.2f}%")
print(f"Fine-Tuned Final Accuracy:  {acc_p2 * 100:.2f}%")
print("=" * 60)