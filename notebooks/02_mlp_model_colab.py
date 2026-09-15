"""
SE4050 - Credit Card Fraud Detection
MLP Baseline Model (Google Colab version)

Reads the shared train/val/test splits from Google Drive
(produced by 01_preprocessing_colab.py) and trains the MLP baseline.

Run this AFTER 01_preprocessing_colab.py has saved the splits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# If Drive isn't already mounted in this session, uncomment:
# from google.colab import drive
# drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/SE4050_fraud_project/data/processed"

# ---------------------------------------------------------
# 1. Load the shared splits
# ---------------------------------------------------------
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

FEATURE_COLS = [c for c in train_df.columns if c != "Class"]

X_train, y_train = train_df[FEATURE_COLS].values, train_df["Class"].values
X_val, y_val = val_df[FEATURE_COLS].values, val_df["Class"].values
X_test, y_test = test_df[FEATURE_COLS].values, test_df["Class"].values

print("Feature columns:", FEATURE_COLS)
print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

# ---------------------------------------------------------
# 2. Class weights (to handle ~0.17% fraud imbalance)
# ---------------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)

# ---------------------------------------------------------
# 3. Build MLP
# ---------------------------------------------------------
def build_mlp(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model

model = build_mlp(X_train.shape[1])
model.summary()

# ---------------------------------------------------------
# 4. Train
# ---------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=8, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1,
)

# ---------------------------------------------------------
# 5. Plot training curves
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("MLP - Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("MLP - Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/../../mlp_training_curves.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 6. Final evaluation on TEST set
# ---------------------------------------------------------
y_proba = model.predict(X_test).ravel()
y_pred_default = (y_proba >= 0.5).astype(int)

print("\n=== Classification Report (threshold=0.5) ===")
print(classification_report(y_test, y_pred_default, target_names=["Normal", "Fraud"]))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

cm = confusion_matrix(y_test, y_pred_default)
print("Confusion Matrix:\n", cm)

# ---------------------------------------------------------
# 7. Threshold analysis
# ---------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recalls, precisions)
print(f"PR-AUC: {pr_auc:.4f}")

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
print(f"Best F1 threshold: {thresholds[best_idx]:.4f}, "
      f"F1={f1_scores[best_idx]:.4f}, "
      f"Precision={precisions[best_idx]:.4f}, Recall={recalls[best_idx]:.4f}")

plt.figure(figsize=(6, 5))
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.plot(thresholds, f1_scores[:-1], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("MLP - Threshold vs Precision/Recall/F1")
plt.legend()
plt.savefig(f"{DATA_DIR}/../../mlp_threshold_analysis.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 8. Save model + results back to Drive
# ---------------------------------------------------------
model.save(f"{DATA_DIR}/../../mlp_fraud_model.keras")
print("\nModel and plots saved to your SE4050_fraud_project Drive folder.")
print("Download them from Drive and push to the GitHub repo.")
