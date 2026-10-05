!pip install -q medmnist

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("Student Roll Number: CH.SC.U4AIE24002")
print("Experiment 10: Convolutional Autoencoder for Medical Image Denoising")
print("=" * 60)

try:
    import medmnist
    from medmnist import PneumoniaMNIST
    dataset_train = PneumoniaMNIST(split="train", download=True)
    dataset_test = PneumoniaMNIST(split="test", download=True)
    X_train = dataset_train.imgs.astype("float32") / 255.0
    X_test = dataset_test.imgs.astype("float32") / 255.0
    print("Loaded PneumoniaMNIST successfully from MedMNIST.")
except Exception as e:
    print("Note: Falling back to MNIST dataset.")
    (X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

noise_factor = 0.3
X_train_noisy = np.clip(X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape), 0.0, 1.0)
X_test_noisy = np.clip(X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape), 0.0, 1.0)

autoencoder = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),
    layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),
    layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")
])

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.fit(
    X_train_noisy, X_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_noisy, X_test),
    verbose=1
)

denoised_images = autoencoder.predict(X_test_noisy[:5], verbose=0)
test_loss = autoencoder.evaluate(X_test_noisy, X_test, verbose=0)
print(f"\n[Roll No: CH.SC.U4AIE24002] Test Reconstruction Loss: {test_loss:.4f}")

n = 5
plt.figure(figsize=(10, 6))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap="gray")
    plt.title(f"Original\n[Roll No: CH.SC.U4AIE24002]" if i == 0 else "Original")
    plt.axis("off")

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(X_test_noisy[i].squeeze(), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_images[i].squeeze(), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()
plt.show()