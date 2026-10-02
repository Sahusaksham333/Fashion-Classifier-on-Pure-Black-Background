"""Train the Fashion-MNIST CNN once and save it for the Streamlit app.

Usage:  python train_model.py
Output: fashion_cnn.keras  (+ metrics.json with test accuracy)
"""
import json
import os

import numpy as np
import pandas as pd
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

DATA_DIR = "input" if os.path.exists("input/fashion-mnist_train.csv") else "."

train = pd.read_csv(os.path.join(DATA_DIR, "fashion-mnist_train.csv")).to_numpy("float32")
test = pd.read_csv(os.path.join(DATA_DIR, "fashion-mnist_test.csv")).to_numpy("float32")

X, y = train[:, 1:] / 255.0, train[:, 0].astype("int32")
X_test, y_test = test[:, 1:] / 255.0, test[:, 0].astype("int32")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12345)
X_train, X_val, X_test = (a.reshape(-1, 28, 28, 1) for a in (X_train, X_val, X_test))

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=256, epochs=30, verbose=2,
          validation_data=(X_val, y_val),
          callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)])

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

model.save("fashion_cnn.keras")
with open("metrics.json", "w") as f:
    json.dump({"test_accuracy": float(acc)}, f)
