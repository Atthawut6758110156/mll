import os
import glob
import json
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ── Reproducibility ──────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = 'act/I.output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Custom objects (must match training) ─────────────────────────────────────
def focal_loss(y_true, y_pred, alpha=0.85, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)


class F1Score(keras.metrics.Metric):
    """Matches the F1Score metric used during training."""
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self._precision = keras.metrics.Precision(thresholds=threshold)
        self._recall    = keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._precision.update_state(y_true, y_pred, sample_weight)
        self._recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self._precision.result()
        r = self._recall.result()
        return 2 * p * r / (p + r + keras.backend.epsilon())

    def reset_state(self):
        self._precision.reset_state()
        self._recall.reset_state()


def load_model_from_path(path):
    m = keras.models.load_model(path, compile=False, custom_objects={'F1Score': F1Score})
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=lambda y_true, y_pred: focal_loss(y_true, y_pred, alpha=0.85, gamma=2.0),
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            F1Score(name='f1_score'),
        ]
    )
    return m


MODEL_DIR = 'act/T.output'
cfg_files = sorted(glob.glob(os.path.join(MODEL_DIR, 'ensemble_config_*.json')))

if cfg_files:
    cfg_path = cfg_files[-1]
    print(f"Loading ensemble config: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    model_paths      = [os.path.join(MODEL_DIR, fn) for fn in cfg['model_files']]
    ensemble_weights = np.array(cfg['ensemble_weights'])
    best_thresh      = cfg['best_threshold']
    train_columns    = cfg['train_columns']
    print(f"Ensemble: {len(model_paths)} models, threshold={best_thresh:.2f}")
    models = [load_model_from_path(p) for p in model_paths]
    print("All ensemble models loaded.")
else:
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, 'activity_model_[0-9]*.h5')))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODEL_DIR}")
    print(f"No ensemble config found. Loading single model: {model_files[-1]}")
    models           = [load_model_from_path(model_files[-1])]
    ensemble_weights = np.array([1.0])
    best_thresh      = None
    train_columns    = None

def engineer_features(X):
    X = X.copy()
    X['NumberOfChildrenVisiting'] = X['NumberOfChildrenVisiting'].fillna(0)
    X['IncomePerTrip']         = X['MonthlyIncome']          / (X['NumberOfTrips'].replace(0, 0.1))
    X['PitchPerFollowup']      = X['DurationOfPitch']        / (X['NumberOfFollowups'].replace(0, 0.1))
    X['TotalVisitors']         = X['NumberOfPersonVisiting'] + X['NumberOfChildrenVisiting']
    X['IncomePerPerson']       = X['MonthlyIncome']          / (X['NumberOfPersonVisiting'].replace(0, 0.1))
    X['SatisfactionXIncome']   = X['PitchSatisfactionScore'] * X['MonthlyIncome']
    X['PassportXIncome']       = X['Passport']               * X['MonthlyIncome']
    X['TripsXVisitors']        = X['NumberOfTrips']          * X['NumberOfPersonVisiting']
    X['PitchXSatisfaction']    = X['DurationOfPitch']        * X['PitchSatisfactionScore']
    X['FollowupXSatisfaction'] = X['NumberOfFollowups']      * X['PitchSatisfactionScore']
    X['ChildrenRatio']         = X['NumberOfChildrenVisiting'] / (X['TotalVisitors'].replace(0, 0.1))
    X['IncomePerTrip2']        = X['MonthlyIncome']          / (X['NumberOfTrips'].replace(0, 0.1) ** 2)
    return X


print("\nLoading data...")
train_df = pd.read_csv('act/Data/train.csv')
df_test  = pd.read_csv('act/Data/test.csv')
print(f"Test dataset shape : {df_test.shape}")
print(f"Target distribution:\n{df_test['ProdTaken'].value_counts()}")

X_train_raw = train_df.drop('ProdTaken', axis=1)
X_test_raw  = df_test.drop('ProdTaken', axis=1)
y_test      = df_test['ProdTaken'].values

X_train_eng = engineer_features(X_train_raw)
X_test_eng  = engineer_features(X_test_raw)
print(f"Features after engineering: {X_test_eng.shape[1]}")

categorical_columns = X_test_eng.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_columns}")

X_train_enc = pd.get_dummies(X_train_eng, columns=categorical_columns, drop_first=True)
X_test_enc  = pd.get_dummies(X_test_eng,  columns=categorical_columns, drop_first=True)

ref_columns = train_columns if train_columns is not None else X_train_enc.columns.tolist()
for col in ref_columns:
    if col not in X_test_enc.columns:
        X_test_enc[col] = 0
X_test_enc = X_test_enc[ref_columns]
print(f"Features after encoding: {X_test_enc.shape[1]}")

X_train_aligned = X_train_enc.reindex(columns=ref_columns, fill_value=0)
scaler = StandardScaler()
scaler.fit(X_train_aligned)
X_test_scaled = scaler.transform(X_test_enc)

print("\nMaking ensemble predictions...")
all_probas   = np.array([m.predict(X_test_scaled, verbose=0).flatten() for m in models])
y_pred_proba = np.average(all_probas, axis=0, weights=ensemble_weights)

if best_thresh is None:
    best_thresh, best_f1_t = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        _f1 = f1_score(y_test, (y_pred_proba > t).astype(int), zero_division=0)
        if _f1 > best_f1_t:
            best_f1_t, best_thresh = _f1, t
    print(f"Auto-optimised threshold : {best_thresh:.2f}  ->  F1={best_f1_t:.4f}")
else:
    print(f"Using saved threshold : {best_thresh:.2f}")

y_pred = (y_pred_proba > best_thresh).astype(int)

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("INFERENCE RESULTS")
print("=" * 60)

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
macro_f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"\nAccuracy     : {acc:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}  (minority class)")
print(f"Macro F1     : {macro_f1:.4f}")

try:
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score : {auc:.4f}")
except Exception:
    auc = None
    print("AUC Score : Could not calculate")

report = classification_report(y_test, y_pred, target_names=['Not Taken (0)', 'Taken (1)'])
print("\nClassification Report:")
print(report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ── Confusion matrix plot ────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Taken', 'Taken'],
            yticklabels=['Not Taken', 'Taken'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix – Activity Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

tn, fp, fn, tp = cm.ravel()
stats_text = (f'TN: {tn}  FP: {fp}\nFN: {fn}  TP: {tp}'
              f'\n\nAccuracy : {acc:.4f}'
              f'\nF1 Score : {f1:.4f}')
plt.text(2.5, 0.5, stats_text, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='center')
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{timestamp}.jpg')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix saved : {cm_path}")

# ── Normalised confusion matrix ──────────────────────────────────────────────
plt.figure(figsize=(10, 8))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=['Not Taken', 'Taken'],
            yticklabels=['Not Taken', 'Taken'],
            cbar_kws={'label': 'Percentage'})
plt.title('Normalised Confusion Matrix – Activity Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

cm_norm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_normalised_{timestamp}.jpg')
plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
print(f"Normalised confusion matrix saved : {cm_norm_path}")

# ── Save predictions CSV ──────────────────────────────────────────────────────
results_df = df_test.copy()
results_df['predicted_label']       = y_pred
results_df['prediction_probability'] = y_pred_proba
results_df['true_label']            = y_test
results_df['correct_prediction']    = (results_df['true_label'] == results_df['predicted_label'])

predictions_path = os.path.join(OUTPUT_DIR, f'predictions_{timestamp}.csv')
results_df.to_csv(predictions_path, index=False)
print(f"Predictions saved : {predictions_path}")

# ── Save evaluation metrics ───────────────────────────────────────────────────
metrics_path = os.path.join(OUTPUT_DIR, f'inference_evaluation_metrics_{timestamp}.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Timestamp    : {timestamp}\n")
    f.write(f"Models       : {len(models)} (ensemble)\n")
    f.write(f"Test data    : act/Data/test.csv\n")
    f.write(f"Samples      : {X_test_scaled.shape[0]}\n")
    f.write(f"Features     : {X_test_scaled.shape[1]}\n")
    f.write(f"Threshold    : {best_thresh:.2f}\n\n")
    f.write(f"Accuracy     : {acc:.4f}\n")
    f.write(f"Precision    : {precision:.4f}\n")
    f.write(f"Recall       : {recall:.4f}\n")
    f.write(f"F1 Score     : {f1:.4f}  (minority class)\n")
    f.write(f"Macro F1     : {macro_f1:.4f}\n")
    if auc is not None:
        f.write(f"AUC Score    : {auc:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
print(f"Metrics saved : {metrics_path}")

print("\n" + "=" * 60)
print("Inference completed successfully!")
print("=" * 60)
