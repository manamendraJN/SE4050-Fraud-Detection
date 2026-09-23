"""
=============================================================================
SE4050 – Deep Learning | Credit Card Fraud Detection
Model: Long Short-Term Memory (LSTM)
Author: Diluka
Dataset: Kaggle Credit Card Fraud Detection
Course: SE4050 – Deep Learning | 2026
=============================================================================

Description:
This script implements a Long Short-Term Memory (LSTM) recurrent neural network
for detecting fraudulent credit card transactions. 

Tabular features (30 features: Time, V1-V28, Amount) are reshaped into 
sequential temporal steps (6 timesteps x 5 features) to simulate sequential
feature groups and allow LSTM gating mechanisms to capture complex interactions.
=============================================================================
"""

import os
import sys
import json
import random
import argparse
import warnings
from pathlib import Path

# Check and guide user if dependencies are not yet installed
MISSING_DEPS = []
try:
    import numpy as np
except ImportError:
    MISSING_DEPS.append("numpy")

try:
    import pandas as pd
except ImportError:
    MISSING_DEPS.append("pandas")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    MISSING_DEPS.append("matplotlib seaborn")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except ImportError:
    MISSING_DEPS.append("tensorflow")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score,
        roc_curve, precision_recall_curve, average_precision_score,
        f1_score, precision_score, recall_score, accuracy_score
    )
    from sklearn.decomposition import PCA
except ImportError:
    MISSING_DEPS.append("scikit-learn")

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    MISSING_DEPS.append("imbalanced-learn")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Global Configurations & Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
TIMESTEPS = 6


def set_seed(seed=SEED):
    """Set random seeds for reproducibility across libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_plotting_theme():
    """Apply sleek dark theme for plots."""
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#e6edf3',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'text.color': '#e6edf3',
        'grid.color': '#21262d',
        'grid.linestyle': '--',
        'font.size': 11
    })


# ---------------------------------------------------------------------------
# Data Loading & Synthetic Fallback
# ---------------------------------------------------------------------------
def load_dataset(data_path='creditcard.csv'):
    """
    Load dataset from csv file. If not present or empty, fallback
    to synthetic dataset matching original schema for demonstration.
    """
    if not os.path.exists(data_path) or os.path.getsize(data_path) < 1000:
        print(f"⚠️ '{data_path}' not found or empty.")
        print("Generating synthetic dataset matching schema for demonstration...")
        print("Tip: Download creditcard.csv from Kaggle for real results.")
        np.random.seed(SEED)
        n_legit, n_fraud = 28000, 492
        legit_feat = np.random.randn(n_legit, 28)
        fraud_feat = np.random.randn(n_fraud, 28) * 1.5 + 0.5
        cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']

        legit_df = pd.DataFrame(np.column_stack([
            np.sort(np.random.uniform(0, 172800, n_legit)),
            legit_feat,
            np.abs(np.random.exponential(50, n_legit)),
            np.zeros(n_legit)
        ]), columns=cols)

        fraud_df = pd.DataFrame(np.column_stack([
            np.sort(np.random.uniform(0, 172800, n_fraud)),
            fraud_feat,
            np.abs(np.random.exponential(120, n_fraud)),
            np.ones(n_fraud)
        ]), columns=cols)

        df = pd.concat([legit_df, fraud_df]).sort_values('Time').reset_index(drop=True)
        df['Class'] = df['Class'].astype(int)
        using_synthetic = True
    else:
        print(f"Loading dataset from '{data_path}'...")
        df = pd.read_csv(data_path)
        using_synthetic = False

    fraud_count = int(df['Class'].sum())
    fraud_pct = 100 * df['Class'].mean()
    time_max_hr = df['Time'].max() / 3600
    print(f"Dataset Shape: {df.shape} | Fraud Samples: {fraud_count:,} ({fraud_pct:.3f}%)")
    print(f"Time Range: {df['Time'].min():.0f}s - {df['Time'].max():.0f}s ({time_max_hr:.1f} hours)")
    return df, using_synthetic


# ---------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
def run_eda(df, output_path='lstm_eda.png'):
    """Perform temporal exploratory data analysis and export visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Temporal Analysis – Credit Card Transactions (Diluka - LSTM)', fontsize=15, fontweight='bold', color='#58a6ff')

    # Transactions over time
    df_temp = df.copy()
    df_temp['Time_hr'] = df_temp['Time'] / 3600
    axes[0, 0].hist(df_temp[df_temp['Class'] == 0]['Time_hr'], bins=48, color='#2ea043', alpha=0.7, label='Legit', density=True)
    axes[0, 0].hist(df_temp[df_temp['Class'] == 1]['Time_hr'], bins=24, color='#f85149', alpha=0.8, label='Fraud', density=True)
    axes[0, 0].set_title('Transaction Volume Over Time (Hours)', color='#58a6ff')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()

    # Fraud by hour-of-day
    df_temp['Hour'] = df_temp['Time_hr'].apply(lambda x: int(x % 24))
    fraud_by_hour = df_temp[df_temp['Class'] == 1].groupby('Hour').size()
    legit_by_hour = df_temp[df_temp['Class'] == 0].groupby('Hour').size()
    fraud_rate_hour = fraud_by_hour / (fraud_by_hour + legit_by_hour.reindex(fraud_by_hour.index, fill_value=0))
    axes[0, 1].bar(fraud_rate_hour.index, fraud_rate_hour.values * 100, color='#f85149', edgecolor='white', alpha=0.8)
    axes[0, 1].set_title('Fraud Rate by Hour of Day', color='#58a6ff')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Fraud Rate (%)')

    # Amount vs Time
    sample_legit = df_temp[df_temp['Class'] == 0].sample(min(500, (df_temp['Class'] == 0).sum()), random_state=SEED)
    sample_fraud = df_temp[df_temp['Class'] == 1]
    axes[1, 0].scatter(sample_legit['Time_hr'], sample_legit['Amount'], alpha=0.3, color='#2ea043', s=5, label='Legit')
    axes[1, 0].scatter(sample_fraud['Time_hr'], sample_fraud['Amount'], alpha=0.8, color='#f85149', s=20, label='Fraud')
    axes[1, 0].set_title('Amount vs Time', color='#58a6ff')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Amount ($)')
    axes[1, 0].set_ylim(0, 500)
    axes[1, 0].legend()

    # Class distribution
    class_counts = df['Class'].value_counts()
    axes[1, 1].pie(class_counts.values, labels=['Legitimate', 'Fraudulent'],
                   colors=['#2ea043', '#f85149'], autopct='%1.3f%%', startangle=90)
    axes[1, 1].set_title('Class Distribution', color='#58a6ff')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 EDA plot saved to '{output_path}'")


# ---------------------------------------------------------------------------
# Preprocessing and Sequence Reshaping
# ---------------------------------------------------------------------------
def preprocess_and_reshape(df, timesteps=TIMESTEPS):
    """
    Standardize features, partition train/val/test splits, apply SMOTE to train,
    and reshape to 3D tensor (samples, timesteps, features_per_step) for LSTM.
    """
    cols_to_drop = [c for c in ['Class', 'Time_hr', 'Hour'] if c in df.columns]
    X = df.drop(cols_to_drop, axis=1).copy()
    y = df['Class'].values

    # Scale Time and Amount
    X['Time'] = StandardScaler().fit_transform(X[['Time']])
    X['Amount'] = StandardScaler().fit_transform(X[['Amount']])
    X_array = X.values

    # Train / Val / Test split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_array, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp
    )

    # SMOTE oversampling on training split only
    print("Applying SMOTE on training partition...")
    smote = SMOTE(random_state=SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    n_features_per_step = X_train_sm.shape[1] // timesteps
    total_features = timesteps * n_features_per_step

    # Pad if features are not cleanly divisible by timesteps
    if X_train_sm.shape[1] != total_features:
        pad = total_features - X_train_sm.shape[1]
        X_train_sm = np.pad(X_train_sm, ((0, 0), (0, pad)), mode='constant')
        X_val = np.pad(X_val, ((0, 0), (0, pad)), mode='constant')
        X_test = np.pad(X_test, ((0, 0), (0, pad)), mode='constant')
        print(f"Padded features from {X.shape[1]} to {total_features}")

    # Reshape to (samples, timesteps, features_per_step)
    X_train_lstm = X_train_sm.reshape(-1, timesteps, n_features_per_step)
    X_val_lstm = X_val.reshape(-1, timesteps, n_features_per_step)
    X_test_lstm = X_test.reshape(-1, timesteps, n_features_per_step)

    print(f"Data Splits after processing:")
    print(f"  Train (SMOTE): {X_train_lstm.shape}, Labels: {y_train_sm.shape}")
    print(f"  Validation:    {X_val_lstm.shape}, Labels: {y_val.shape}")
    print(f"  Test:          {X_test_lstm.shape}, Labels: {y_test.shape}")

    return (X_train_lstm, y_train_sm), (X_val_lstm, y_val), (X_test_lstm, y_test), timesteps, n_features_per_step


# ---------------------------------------------------------------------------
# LSTM Architecture Definition
# ---------------------------------------------------------------------------
def build_lstm(timesteps, n_features, dropout=0.3, l2_reg=1e-4):
    """
    Construct stacked LSTM model for credit card fraud detection.
    
    Structure:
      Input (timesteps, n_features)
      -> LSTM(64, return_sequences=True) + BatchNorm + Dropout(0.3)
      -> LSTM(32, return_sequences=False) + BatchNorm + Dropout(0.3)
      -> Dense(16, relu) + Dropout(0.2)
      -> Dense(1, sigmoid)
    """
    inp = layers.Input(shape=(timesteps, n_features), name='lstm_input')

    # First LSTM layer: returns sequence for stacking
    x = layers.LSTM(
        64, return_sequences=True,
        kernel_regularizer=regularizers.l2(l2_reg),
        recurrent_regularizer=regularizers.l2(l2_reg),
        name='lstm_1'
    )(inp)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(dropout, name='dropout_1')(x)

    # Second LSTM layer: outputs final hidden state
    x = layers.LSTM(
        32,
        kernel_regularizer=regularizers.l2(l2_reg),
        recurrent_regularizer=regularizers.l2(l2_reg),
        name='lstm_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(dropout, name='dropout_2')(x)

    # Dense classification head
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='dense_1')(x)
    x = layers.Dropout(0.2, name='dropout_3')(x)

    # Sigmoid output for fraud probability
    out = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs=inp, outputs=out, name='LSTM_FraudDetector')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    return model


def draw_architecture_diagram(timesteps, n_features, output_path='lstm_architecture.png'):
    """Generate architecture block diagram."""
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 9)
    ax.axis('off')
    ax.set_title('LSTM Architecture – Credit Card Fraud Detection (Diluka)', fontsize=14, fontweight='bold', color='#58a6ff')

    layers_vis = [
        (0.5, 'Input', f'(batch, {timesteps} timesteps, {n_features} features)', '#1f6feb'),
        (2.0, 'LSTM(64) + BatchNorm + Dropout(0.3)', 'return_sequences=True | L2 regularized', '#7d2ae8'),
        (3.5, 'LSTM(32) + BatchNorm + Dropout(0.3)', 'return_sequences=False | Hidden state output', '#7d2ae8'),
        (5.0, 'Dense(16) + ReLU + Dropout(0.2)', 'Dense classification head', '#238636'),
        (6.5, 'Dense(1) + Sigmoid', 'Output: P(Fraud in [0,1])', '#f85149'),
    ]

    for y_pos, title, subtitle, color in layers_vis:
        rect = plt.Rectangle((1.0, y_pos - 0.38), 8, 0.76, color=color, alpha=0.85, zorder=3)
        ax.add_patch(rect)
        ax.text(5, y_pos + 0.08, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color='white', zorder=4)
        ax.text(5, y_pos - 0.18, subtitle, ha='center', va='center', fontsize=8.5, color='#e6edf3', alpha=0.9, zorder=4)

    for i in range(len(layers_vis) - 1):
        y1, y2 = layers_vis[i][0] + 0.38, layers_vis[i + 1][0] - 0.38
        ax.annotate('', xy=(5, y2), xytext=(5, y1), arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📐 Architecture diagram saved to '{output_path}'")


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
def train_lstm_model(model, train_data, val_data, epochs=100, batch_size=256, best_model_path='best_lstm_model.keras'):
    """Train LSTM with early stopping and learning rate scheduling."""
    X_train_lstm, y_train_sm = train_data
    X_val_lstm, y_val = val_data

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_model_path, monitor='val_auc', mode='max', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    print(f"\n🚀 Training LSTM Model ({epochs} max epochs, batch_size={batch_size})...")
    history = model.fit(
        X_train_lstm, y_train_sm,
        validation_data=(X_val_lstm, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    best_epoch = int(np.argmax(history.history['val_auc']) + 1)
    print(f"✅ Training complete! Best epoch: {best_epoch}")
    return history


def plot_training_history(history, output_path='lstm_training_history.png'):
    """Plot Loss, Accuracy, ROC-AUC, and Recall curves across epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('LSTM Training History (Diluka)', fontsize=16, fontweight='bold', color='#58a6ff')

    metric_pairs = [
        ('loss', 'val_loss', 'Loss', '#f85149', '#ff7b72'),
        ('accuracy', 'val_accuracy', 'Accuracy', '#3fb950', '#56d364'),
        ('auc', 'val_auc', 'ROC-AUC', '#58a6ff', '#79c0ff'),
        ('recall', 'val_recall', 'Recall', '#d2a8ff', '#e6c6ff'),
    ]

    best_ep = int(np.argmax(history.history['val_auc']) + 1)

    for ax, (tk, vk, title, c1, c2) in zip(axes.flatten(), metric_pairs):
        ep = range(1, len(history.history[tk]) + 1)
        ax.plot(ep, history.history[tk], color=c1, linewidth=2, label='Train')
        ax.plot(ep, history.history[vk], color=c2, linewidth=2, linestyle='--', label='Val')
        ax.axvline(best_ep, color='#f0883e', linestyle=':', linewidth=1.5, label=f'Best (ep {best_ep})')
        ax.set_title(title, color='#e6edf3')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Training curves saved to '{output_path}'")


# ---------------------------------------------------------------------------
# Threshold Optimization & Test Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, val_data, test_data, output_eval_plot='lstm_evaluation.png'):
    """
    Find optimal classification threshold on validation set (maximizing F1),
    then evaluate on unseen test set and plot confusion matrix, ROC, PR curves.
    """
    X_val_lstm, y_val = val_data
    X_test_lstm, y_test = test_data

    # Tune threshold on validation set
    y_val_prob = model.predict(X_val_lstm, verbose=0).flatten()
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [f1_score(y_val, (y_val_prob >= t).astype(int)) for t in thresholds]
    optimal_thr = float(thresholds[np.argmax(f1_scores)])
    print(f"\n🎯 Optimal validation threshold: {optimal_thr:.2f} | Best val F1: {max(f1_scores):.4f}")

    # Evaluate on test set
    y_test_prob = model.predict(X_test_lstm, verbose=0).flatten()
    y_test_pred = (y_test_prob >= optimal_thr).astype(int)

    acc = float(accuracy_score(y_test, y_test_pred))
    prec = float(precision_score(y_test, y_test_pred, zero_division=0))
    rec = float(recall_score(y_test, y_test_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_test_prob))

    print('=' * 55)
    print('       LSTM – TEST SET RESULTS (Author: Diluka)')
    print('=' * 55)
    print(f'  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1-Score  : {f1:.4f}')
    print(f'  ROC-AUC   : {auc:.4f}')
    print('=' * 55)
    print('\n' + classification_report(y_test, y_test_pred, target_names=['Legitimate', 'Fraudulent'], zero_division=0))

    # Evaluation plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('LSTM – Evaluation Results (Diluka)', fontsize=16, fontweight='bold', color='#58a6ff')

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'], ax=axes[0])
    axes[0].set_title('Confusion Matrix', color='#58a6ff')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    axes[1].plot(fpr, tpr, color='#7d2ae8', linewidth=2.5, label=f'AUC = {auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].fill_between(fpr, tpr, alpha=0.2, color='#7d2ae8')
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].set_title('ROC Curve', color='#58a6ff')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    pc, rc, _ = precision_recall_curve(y_test, y_test_prob)
    ap = average_precision_score(y_test, y_test_prob)
    axes[2].plot(rc, pc, color='#7d2ae8', linewidth=2.5, label=f'AP = {ap:.4f}')
    axes[2].fill_between(rc, pc, alpha=0.2, color='#7d2ae8')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve', color='#58a6ff')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_eval_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Evaluation plots saved to '{output_eval_plot}'")

    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'optimal_threshold': optimal_thr
    }
    return results, y_test_pred, y_test_prob


# ---------------------------------------------------------------------------
# Hidden State Interpretability (PCA)
# ---------------------------------------------------------------------------
def analyze_hidden_states(model, test_data, output_path='lstm_hidden_states.png'):
    """Extract intermediate hidden states from LSTM and visualize representations via PCA."""
    X_test_lstm, y_test = test_data
    intermediate_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('lstm_2').output
    )

    n_samples = min(500, len(X_test_lstm))
    hidden_states = intermediate_model.predict(X_test_lstm[:n_samples], verbose=0)
    y_test_sample = y_test[:n_samples]

    pca = PCA(n_components=2, random_state=SEED)
    hs_2d = pca.fit_transform(hidden_states)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(hs_2d[:, 0], hs_2d[:, 1],
                          c=y_test_sample, cmap='RdYlGn_r', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Class (0=Legit, 1=Fraud)')
    plt.title('LSTM Hidden State Visualization (PCA 2D)\nFraud vs Legitimate Transaction Representations (Diluka)',
              color='#58a6ff', fontsize=13)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🔬 Hidden states visualization saved to '{output_path}'")


# ---------------------------------------------------------------------------
# Save Results & Artifacts
# ---------------------------------------------------------------------------
def save_artifacts(model, metrics, timesteps, n_features, out_dir='.'):
    """Save trained model, JSON summary, and CSV metric table."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_save_path = out_path / 'lstm_fraud_model.keras'
    model.save(str(model_save_path))

    # Save JSON summary
    summary = {
        'model': 'LSTM',
        'author': 'Diluka',
        'accuracy': round(metrics['accuracy'], 4),
        'precision': round(metrics['precision'], 4),
        'recall': round(metrics['recall'], 4),
        'f1': round(metrics['f1'], 4),
        'roc_auc': round(metrics['roc_auc'], 4),
        'optimal_threshold': round(float(metrics['optimal_threshold']), 2),
        'architecture': f'({timesteps},{n_features})->LSTM(64)->LSTM(32)->Dense(16)->1',
        'params': int(model.count_params())
    }
    json_path = out_path / 'lstm_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save CSV metrics
    csv_path = out_path / 'lstm_results.csv'
    pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
        'Test': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['roc_auc']]
    }).round(4).to_csv(csv_path, index=False)

    print(f"\n💾 Model and evaluation artifacts saved:")
    print(f"  Model  : {model_save_path}")
    print(f"  Summary: {json_path}")
    print(f"  Metrics: {csv_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection using LSTM (Diluka's Part)")
    parser.add_argument('--data', type=str, default='creditcard.csv', help='Path to creditcard.csv dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 256)')
    parser.add_argument('--out_dir', type=str, default='.', help='Directory to store output artifacts')
    parser.add_argument('--no_plots', action='store_true', help='Skip plotting visualizations')

    args = parser.parse_args()

    if MISSING_DEPS:
        print("❌ Error: Missing required dependencies to execute training pipeline:")
        for dep in MISSING_DEPS:
            print(f"   - {dep}")
        print("\nPlease install the project dependencies by running:")
        print("   pip install -r requirements.txt\n")
        sys.exit(1)

    set_seed(SEED)
    setup_plotting_theme()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SE4050 Deep Learning – Credit Card Fraud Detection")
    print("Author : Diluka | Model: Long Short-Term Memory (LSTM)")
    print("=" * 60)
    print(f"TensorFlow Version: {tf.__version__} | GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # 1. Load Data
    df, is_synthetic = load_dataset(args.data)

    # 2. EDA
    if not args.no_plots:
        run_eda(df, output_path=str(out_dir / 'lstm_eda.png'))

    # 3. Preprocessing
    train_data, val_data, test_data, timesteps, n_features = preprocess_and_reshape(df, timesteps=TIMESTEPS)

    # 4. Model Creation & Architecture Diagram
    model = build_lstm(timesteps=timesteps, n_features=n_features)
    model.summary()

    if not args.no_plots:
        draw_architecture_diagram(timesteps, n_features, output_path=str(out_dir / 'lstm_architecture.png'))

    # 5. Training
    best_model_path = str(out_dir / 'best_lstm_model.keras')
    history = train_lstm_model(model, train_data, val_data, epochs=args.epochs, batch_size=args.batch_size, best_model_path=best_model_path)

    if not args.no_plots:
        plot_training_history(history, output_path=str(out_dir / 'lstm_training_history.png'))

    # 6. Evaluation
    metrics, y_pred, y_prob = evaluate_model(
        model, val_data, test_data,
        output_eval_plot=str(out_dir / 'lstm_evaluation.png')
    )

    # 7. Hidden State Representation
    if not args.no_plots:
        analyze_hidden_states(model, test_data, output_path=str(out_dir / 'lstm_hidden_states.png'))

    # 8. Save Artifacts
    save_artifacts(model, metrics, timesteps, n_features, out_dir=args.out_dir)
    print("\n🎉 Diluka's LSTM Pipeline executed successfully!")


if __name__ == '__main__':
    main()
