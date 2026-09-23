"""
Credit Card Fraud Detection using LSTM (Long Short-Term Memory)
Full Python Script converted from LSTM (1).ipynb with all cells and bug fixes.
"""

import os
import random
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ============================================================
# CELL 1: Setup & Reproducibility
# ============================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
    'font.size': 11,
})

print(f"TensorFlow: {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("✅ Setup complete!\n")


def main():
    # ============================================================
    # CELL 2: Dataset Loading
    # ============================================================
    print("=" * 60)
    print("STEP 1: Dataset Loading")
    print("=" * 60)

    csv_path = 'creditcard.csv'
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 1000:
        print("Dataset 'creditcard.csv' not found or empty.")
        print("Generating synthetic dataset for demonstration...")
        print("(For real benchmark results, place Kaggle's creditcard.csv in this directory.)")
        np.random.seed(SEED)
        n_legit, n_fraud = 28000, 492
        legit_feat = np.random.randn(n_legit, 28)
        fraud_feat = np.random.randn(n_fraud, 28) * 1.5 + 0.5
        cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']

        legit_df = pd.DataFrame(
            np.column_stack([
                np.sort(np.random.uniform(0, 172800, n_legit)),
                legit_feat,
                np.abs(np.random.exponential(50, n_legit)),
                np.zeros(n_legit),
            ]),
            columns=cols,
        )
        fraud_df = pd.DataFrame(
            np.column_stack([
                np.sort(np.random.uniform(0, 172800, n_fraud)),
                fraud_feat,
                np.abs(np.random.exponential(120, n_fraud)),
                np.ones(n_fraud),
            ]),
            columns=cols,
        )
        df = pd.concat([legit_df, fraud_df]).sort_values('Time').reset_index(drop=True)
        df['Class'] = df['Class'].astype(int)
        using_synthetic = True
    else:
        print(f"Loading '{csv_path}' from disk...")
        df = pd.read_csv(csv_path)
        using_synthetic = False

    print(f"Dataset shape : {df.shape}")
    print(f"Fraud cases   : {df['Class'].sum():,} ({100 * df['Class'].mean():.3f}%)")
    print(f"Time range    : {df['Time'].min():.0f}s – {df['Time'].max():.0f}s ({df['Time'].max() / 3600:.1f} hours)\n")

    # ============================================================
    # CELL 3: EDA – Temporal Analysis
    # ============================================================
    print("=" * 60)
    print("STEP 2: Exploratory Data Analysis (EDA)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Temporal Analysis – Credit Card Transactions', fontsize=15, fontweight='bold', color='#58a6ff')

    # Transactions over time
    df['Time_hr'] = df['Time'] / 3600
    axes[0, 0].hist(df[df['Class'] == 0]['Time_hr'], bins=48, color='#2ea043', alpha=0.7, label='Legit', density=True)
    axes[0, 0].hist(df[df['Class'] == 1]['Time_hr'], bins=24, color='#f85149', alpha=0.8, label='Fraud', density=True)
    axes[0, 0].set_title('Transaction Volume Over Time (Hours)', color='#58a6ff')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()

    # Fraud by hour-of-day (modulo 24h assuming 2-day window)
    df['Hour'] = df['Time_hr'].apply(lambda x: int(x % 24))
    fraud_by_hour = df[df['Class'] == 1].groupby('Hour').size()
    legit_by_hour = df[df['Class'] == 0].groupby('Hour').size()
    fraud_rate_hour = fraud_by_hour / (fraud_by_hour + legit_by_hour.reindex(fraud_by_hour.index, fill_value=0))
    axes[0, 1].bar(fraud_rate_hour.index, fraud_rate_hour.values * 100, color='#f85149', edgecolor='white', alpha=0.8)
    axes[0, 1].set_title('Fraud Rate by Hour of Day', color='#58a6ff')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Fraud Rate (%)')

    # Amount vs Time (fraud vs legit)
    sample_legit = df[df['Class'] == 0].sample(min(500, (df['Class'] == 0).sum()), random_state=SEED)
    sample_fraud = df[df['Class'] == 1]
    axes[1, 0].scatter(sample_legit['Time_hr'], sample_legit['Amount'], alpha=0.3, color='#2ea043', s=5, label='Legit')
    axes[1, 0].scatter(sample_fraud['Time_hr'], sample_fraud['Amount'], alpha=0.8, color='#f85149', s=20, label='Fraud')
    axes[1, 0].set_title('Amount vs Time', color='#58a6ff')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Amount ($)')
    axes[1, 0].set_ylim(0, 500)
    axes[1, 0].legend()

    # Class distribution
    class_counts = df['Class'].value_counts()
    axes[1, 1].pie(
        class_counts.values,
        labels=['Legitimate', 'Fraudulent'],
        colors=['#2ea043', '#f85149'],
        autopct='%1.3f%%',
        startangle=90,
    )
    axes[1, 1].set_title('Class Distribution', color='#58a6ff')

    plt.tight_layout()
    plt.savefig('lstm_eda.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 'lstm_eda.png'\n")

    # ============================================================
    # CELL 4: Preprocessing and Sequence Reshaping
    # ============================================================
    print("=" * 60)
    print("STEP 3: Preprocessing & Data Splitting")
    print("=" * 60)

    # Feature scaling
    X = df.drop(['Class', 'Time_hr', 'Hour'], axis=1).copy()
    y = df['Class'].values

    X['Time'] = StandardScaler().fit_transform(X[['Time']])
    X['Amount'] = StandardScaler().fit_transform(X[['Amount']])
    X_array = X.values

    # Train/Val/Test split (70/15/15) BEFORE SMOTE
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_array, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp
    )

    # Apply SMOTE to training set only
    smote = SMOTE(random_state=SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print("Splits:")
    print(f"  Train (after SMOTE): {X_train_sm.shape}")
    print(f"  Val                : {X_val.shape}")
    print(f"  Test               : {X_test.shape}\n")

    # ============================================================
    # CELL 5: Reshape data for LSTM input
    # ============================================================
    print("=" * 60)
    print("STEP 4: Reshape Data for LSTM Input")
    print("=" * 60)

    TIMESTEPS = 6
    N_FEATURES_PER_STEP = X_train_sm.shape[1] // TIMESTEPS

    # Ensure divisibility
    total_features = TIMESTEPS * N_FEATURES_PER_STEP
    if X_train_sm.shape[1] != total_features:
        pad = total_features - X_train_sm.shape[1]
        X_train_sm = np.pad(X_train_sm, ((0, 0), (0, pad)), mode='constant')
        X_val = np.pad(X_val, ((0, 0), (0, pad)), mode='constant')
        X_test = np.pad(X_test, ((0, 0), (0, pad)), mode='constant')
        print(f"Padded features from {X.shape[1]} to {total_features}")

    X_train_lstm = X_train_sm.reshape(-1, TIMESTEPS, N_FEATURES_PER_STEP)
    X_val_lstm = X_val.reshape(-1, TIMESTEPS, N_FEATURES_PER_STEP)
    X_test_lstm = X_test.reshape(-1, TIMESTEPS, N_FEATURES_PER_STEP)

    print(f"LSTM input shape: (samples, timesteps={TIMESTEPS}, features/step={N_FEATURES_PER_STEP})")
    print(f"Train: {X_train_lstm.shape} | Val: {X_val_lstm.shape} | Test: {X_test_lstm.shape}\n")

    # ============================================================
    # CELL 6: Build LSTM model
    # ============================================================
    print("=" * 60)
    print("STEP 5: Build LSTM Architecture")
    print("=" * 60)

    def build_lstm(timesteps, n_features, dropout=0.3, l2_reg=1e-4):
        model = keras.Sequential([
            layers.Input(shape=(timesteps, n_features), name='lstm_input'),

            # First LSTM layer – return sequences for stacking
            layers.LSTM(
                64,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg),
                name='lstm_1',
            ),
            layers.BatchNormalization(name='bn_1'),
            layers.Dropout(dropout, name='dropout_1'),

            # Second LSTM layer – outputs final hidden state only
            layers.LSTM(
                32,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg),
                name='lstm_2',
            ),
            layers.BatchNormalization(name='bn_2'),
            layers.Dropout(dropout, name='dropout_2'),

            # Dense classification head
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='dense_1'),
            layers.Dropout(0.2, name='dropout_3'),

            # Output
            layers.Dense(1, activation='sigmoid', name='output'),
        ], name='LSTM_FraudDetector')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ],
        )
        return model

    lstm_model = build_lstm(TIMESTEPS, N_FEATURES_PER_STEP)
    lstm_model.summary()
    print()

    # ============================================================
    # CELL 7: Architecture visualization
    # ============================================================
    print("=" * 60)
    print("STEP 6: Architecture Diagram")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 9)
    ax.axis('off')
    ax.set_title('LSTM Architecture – Credit Card Fraud Detection', fontsize=14, fontweight='bold', color='#58a6ff')

    layers_vis = [
        (0.5, 'Input', f'(batch, {TIMESTEPS} timesteps, {N_FEATURES_PER_STEP} features)', '#1f6feb'),
        (2.0, 'LSTM(64) + BatchNorm + Dropout(0.3)', 'return_sequences=True | L2 regularized', '#7d2ae8'),
        (3.5, 'LSTM(32) + BatchNorm + Dropout(0.3)', 'return_sequences=False | Hidden state output', '#7d2ae8'),
        (5.0, 'Dense(16) + ReLU + Dropout(0.2)', 'Dense classification head', '#238636'),
        (6.5, 'Dense(1) + Sigmoid', 'Output: P(Fraud ∈ [0,1])', '#f85149'),
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
    plt.savefig('lstm_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 'lstm_architecture.png'\n")

    # ============================================================
    # CELL 8: Training
    # ============================================================
    print("=" * 60)
    print("STEP 7: Model Training")
    print("=" * 60)

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_lstm_model.keras', monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    print("🚀 Training LSTM...")
    history = lstm_model.fit(
        X_train_lstm,
        y_train_sm,
        validation_data=(X_val_lstm, y_val),
        epochs=100,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
    )
    best_epoch = int(np.argmax(history.history['val_auc']) + 1)
    print(f"\n✅ Training complete! Best epoch: {best_epoch}\n")

    # ============================================================
    # CELL 9: Training History Visualization
    # ============================================================
    print("=" * 60)
    print("STEP 8: Plot Training History")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('LSTM Training History', fontsize=16, fontweight='bold', color='#58a6ff')

    metric_pairs = [
        ('loss', 'val_loss', 'Loss', '#f85149', '#ff7b72'),
        ('accuracy', 'val_accuracy', 'Accuracy', '#3fb950', '#56d364'),
        ('auc', 'val_auc', 'ROC-AUC', '#58a6ff', '#79c0ff'),
        ('recall', 'val_recall', 'Recall', '#d2a8ff', '#e6c6ff'),
    ]

    for ax, (tk, vk, title, c1, c2) in zip(axes.flatten(), metric_pairs):
        ep = range(1, len(history.history[tk]) + 1)
        ax.plot(ep, history.history[tk], color=c1, linewidth=2, label='Train')
        ax.plot(ep, history.history[vk], color=c2, linewidth=2, linestyle='--', label='Val')
        best_ep = np.argmax(history.history['val_auc']) + 1
        ax.axvline(best_ep, color='#f0883e', linestyle=':', linewidth=1.5, label=f'Best (ep {best_ep})')
        ax.set_title(title, color='#e6edf3')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lstm_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 'lstm_training_history.png'\n")

    # ============================================================
    # CELL 10: Threshold Tuning + Final Test Evaluation
    # ============================================================
    print("=" * 60)
    print("STEP 9: Threshold Tuning & Test Evaluation")
    print("=" * 60)

    y_val_prob = lstm_model.predict(X_val_lstm, verbose=0).flatten()
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [f1_score(y_val, (y_val_prob >= t).astype(int)) for t in thresholds]
    optimal_thr = thresholds[np.argmax(f1_scores)]
    print(f"Optimal threshold: {optimal_thr:.2f} | Best val F1: {max(f1_scores):.4f}")

    # Test evaluation
    y_test_prob = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    y_test_pred = (y_test_prob >= optimal_thr).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_prob)

    print('=' * 55)
    print('       LSTM – TEST SET RESULTS')
    print('=' * 55)
    print(f'  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1-Score  : {f1:.4f}')
    print(f'  ROC-AUC   : {auc:.4f}')
    print('=' * 55)
    print()
    print(classification_report(y_test, y_test_pred, target_names=['Legitimate', 'Fraudulent']))

    # ============================================================
    # CELL 11: Evaluation Visualizations
    # ============================================================
    print("=" * 60)
    print("STEP 10: Evaluation Visualizations")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('LSTM – Evaluation Results', fontsize=16, fontweight='bold', color='#58a6ff')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Purples',
        xticklabels=['Legit', 'Fraud'],
        yticklabels=['Legit', 'Fraud'],
        ax=axes[0],
    )
    axes[0].set_title('Confusion Matrix', color='#58a6ff')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    axes[1].plot(fpr, tpr, color='#7d2ae8', linewidth=2.5, label=f'AUC = {auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].fill_between(fpr, tpr, alpha=0.2, color='#7d2ae8')
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].set_title('ROC Curve', color='#58a6ff')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Precision-Recall Curve
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
    plt.savefig('lstm_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 'lstm_evaluation.png'\n")

    # ============================================================
    # CELL 12: Hidden State Analysis (Corrected & Robust)
    # ============================================================
    print("=" * 60)
    print("STEP 11: Hidden State Analysis (LSTM Interpretability)")
    print("=" * 60)

    n_samples = min(500, len(X_test_lstm))
    y_test_sample = y_test[:n_samples]

    try:
        # In Keras 3 / TensorFlow 2.16+, use lstm_model.inputs (plural)
        lstm_layer_output_model = tf.keras.Model(
            inputs=lstm_model.inputs,
            outputs=lstm_model.get_layer('lstm_2').output,
        )
        hidden_states = lstm_layer_output_model.predict(X_test_lstm[:n_samples], verbose=0)
    except Exception as e:
        print(f"Notice: Using direct layer forward pass fallback ({e})")
        # Universal fallback: sequential forward pass through layers up to 'lstm_2'
        x_sub = tf.convert_to_tensor(X_test_lstm[:n_samples], dtype=tf.float32)
        for layer in lstm_model.layers:
            x_sub = layer(x_sub, training=False)
            if layer.name == 'lstm_2':
                break
        hidden_states = np.array(x_sub)

    pca = PCA(n_components=2, random_state=SEED)
    hs_2d = pca.fit_transform(hidden_states)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        hs_2d[:, 0],
        hs_2d[:, 1],
        c=y_test_sample,
        cmap='RdYlGn_r',
        alpha=0.6,
        s=30,
    )
    plt.colorbar(scatter, label='Class (0=Legit, 1=Fraud)')
    plt.title(
        'LSTM Hidden State Visualization (PCA 2D)\nFraud vs Legitimate Transaction Representations',
        color='#58a6ff',
        fontsize=13,
    )
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_hidden_states.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 'lstm_hidden_states.png'\n")

    # ============================================================
    # CELL 13: Save Results & Summary
    # ============================================================
    print("=" * 60)
    print("STEP 12: Saving Results and Summary")
    print("=" * 60)

    lstm_model.save('lstm_fraud_model.keras')

    summary = {
        'model': 'LSTM',
        'author': 'Diluka',
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'roc_auc': round(float(auc), 4),
        'optimal_threshold': round(float(optimal_thr), 2),
        'architecture': f'({TIMESTEPS},{N_FEATURES_PER_STEP})→LSTM(64)→LSTM(32)→Dense(16)→1',
        'params': int(lstm_model.count_params()),
    }
    with open('lstm_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
        'Test': [round(float(acc), 4), round(float(prec), 4), round(float(rec), 4), round(float(f1), 4), round(float(auc), 4)],
    })
    results_df.to_csv('lstm_results.csv', index=False)

    print("✅ LSTM model and results saved!")
    print("\n📊 Final Results Summary:")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")


if __name__ == '__main__':
    main()
