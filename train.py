import os
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime

tf.random.set_seed(42)
np.random.seed(42)

# ── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = 'act/T.output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Load training data ───────────────────────────────────────────────────────
print("Loading training data...")
train_df = pd.read_csv('act/Data/train.csv')

print(f"Train shape   : {train_df.shape}")
print(f"Target dist   :\n{train_df['ProdTaken'].value_counts()}")

# ── Features / target ───────────────────────────────────────────────────────
X = train_df.drop('ProdTaken', axis=1)
y = train_df['ProdTaken'].values

# ── Feature engineering ─────────────────────────────────────────────────────
X['NumberOfChildrenVisiting'] = X['NumberOfChildrenVisiting'].fillna(0)
X['IncomePerTrip']        = X['MonthlyIncome']          / (X['NumberOfTrips'].replace(0, 0.1))
X['PitchPerFollowup']     = X['DurationOfPitch']        / (X['NumberOfFollowups'].replace(0, 0.1))
X['TotalVisitors']        = X['NumberOfPersonVisiting'] + X['NumberOfChildrenVisiting']
X['IncomePerPerson']      = X['MonthlyIncome']          / (X['NumberOfPersonVisiting'].replace(0, 0.1))
X['SatisfactionXIncome']  = X['PitchSatisfactionScore'] * X['MonthlyIncome']
# Extended interactions
X['PassportXIncome']      = X['Passport']               * X['MonthlyIncome']
X['TripsXVisitors']       = X['NumberOfTrips']          * X['NumberOfPersonVisiting']
X['PitchXSatisfaction']   = X['DurationOfPitch']        * X['PitchSatisfactionScore']
X['FollowupXSatisfaction']= X['NumberOfFollowups']      * X['PitchSatisfactionScore']
X['ChildrenRatio']        = X['NumberOfChildrenVisiting'] / (X['TotalVisitors'].replace(0, 0.1))
X['IncomePerTrip2']       = X['MonthlyIncome']          / (X['NumberOfTrips'].replace(0, 0.1) ** 2)
print(f"Features after engineering: {X.shape[1]}")

# ── One-hot encode categorical columns ──────────────────────────────────────
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
print(f"Features after encoding: {X_encoded.shape[1]}")

# ── Train / validation split (stratified 80/20) ─────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain samples : {len(X_tr)}")
print(f"Val   samples : {len(X_val)}")

# ── Scale features (fit on train only) ───────────────────────────────────────
scaler = StandardScaler()
X_scaled     = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

# ── Class weights ─────────────────────────────────────────────────────────────
# Higher weight compensates for class imbalance without oversampling
class_weight_dict = {0: 1.0, 1: 5.0}
print(f"Class weights   : {class_weight_dict}")

# ── Custom loss functions ────────────────────────────────────────────────────
def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2.0):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weights = y_true * (pos_weight - 1.0) + 1.0
    return tf.reduce_mean(bce * weights)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)


def combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.01):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])
        return alpha * bce_loss + beta * l1_reg
    return loss


class F1Score(keras.metrics.Metric):
    """Differentiable-compatible F1 metric tracked during training."""
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


def compile_model_with_custom_loss(model, loss_function, optimizer='adam', metrics=None):
    if metrics is None:
        metrics = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            F1Score(name='f1_score'),
        ]
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    return model

# ── Build model factory (functional API with skip connections) ───────────────
def build_model(input_dim):
    """Wide residual-style network with skip connections + L2 + dropout."""
    reg = keras.regularizers.l2(1e-4)

    inputs = keras.Input(shape=(input_dim,))

    # Block 1
    x = keras.layers.Dense(256, kernel_regularizer=reg)(inputs)
    x = keras.layers.LeakyReLU(negative_slope=0.01)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    # Block 2 with skip
    h = keras.layers.Dense(256, kernel_regularizer=reg)(x)
    h = keras.layers.LeakyReLU(negative_slope=0.01)(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Dropout(0.25)(h)
    x = keras.layers.Add()([x, h])          # residual skip

    # Block 3
    x = keras.layers.Dense(128, kernel_regularizer=reg)(x)
    x = keras.layers.LeakyReLU(negative_slope=0.01)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.20)(x)

    # Block 4 with skip
    h = keras.layers.Dense(128, kernel_regularizer=reg)(x)
    h = keras.layers.LeakyReLU(negative_slope=0.01)(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Dropout(0.20)(h)
    x = keras.layers.Add()([x, h])          # residual skip

    # Block 5
    x = keras.layers.Dense(64, kernel_regularizer=reg)(x)
    x = keras.layers.LeakyReLU(negative_slope=0.01)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.15)(x)

    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    m = keras.Model(inputs, outputs)
    compile_model_with_custom_loss(
        m,
        lambda y_true, y_pred: focal_loss(y_true, y_pred, alpha=0.85, gamma=2.0),
        optimizer=keras.optimizers.Adam(learning_rate=5e-4)
    )
    return m


# ── Multi-restart: train N models, ensemble all predictions ──────────────────
N_RESTARTS      = 4
all_models      = []
all_val_probas  = []

for run in range(1, N_RESTARTS + 1):
    print(f"\n{'='*50}")
    print(f"Training run {run}/{N_RESTARTS}")
    print('='*50)
    tf.random.set_seed(run * 7)
    np.random.seed(run * 13)
    m = build_model(X_scaled.shape[1])
    m.fit(
        X_scaled, y_tr,
        epochs=400,
        batch_size=64,
        validation_data=(X_val_scaled, y_val),
        class_weight=class_weight_dict,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_f1_score',
                patience=50,
                restore_best_weights=True,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score',
                factor=0.5,
                patience=15,
                min_lr=1e-6,
                mode='max',
                verbose=0
            ),
        ]
    )
    val_proba = m.predict(X_val_scaled, verbose=0).flatten()
    run_f1 = max(
        f1_score(y_val, (val_proba > t).astype(int), zero_division=0)
        for t in np.arange(0.10, 0.91, 0.01)
    )
    print(f"Run {run} best val-F1 = {run_f1:.4f}")
    all_models.append(m)
    all_val_probas.append(val_proba)

# ── Ensemble: weighted average by each model's val F1 ────────────────────────
run_f1s = [
    max(f1_score(y_val, (p > t).astype(int), zero_division=0)
        for t in np.arange(0.10, 0.91, 0.01))
    for p in all_val_probas
]
weights = np.array(run_f1s) ** 2          # square to up-weight better models
weights /= weights.sum()
print(f"\nEnsemble weights: {[f'{w:.3f}' for w in weights]}")
best_val_proba = sum(w * p for w, p in zip(weights, all_val_probas))
best_model     = all_models[int(np.argmax(run_f1s))]
model          = best_model
print(f"Best single-model val-F1 : {max(run_f1s):.4f}")

# ── Evaluate ensemble on validation set ──────────────────────────────────────
print("\nEvaluating ensemble on validation set...")
loss, acc_keras, prec_keras, rec_keras, f1_keras = model.evaluate(X_val_scaled, y_val, verbose=0)
print(f"Best single model (threshold=0.5):")
print(f"  Accuracy  : {acc_keras:.4f}")
print(f"  Precision : {prec_keras:.4f}")
print(f"  Recall    : {rec_keras:.4f}")
print(f"  F1 Score  : {f1_keras:.4f}")

# ── Find threshold that maximises ensemble val F1 ────────────────────────────
y_pred_proba = best_val_proba
best_thresh, best_f1 = 0.5, 0.0
for t in np.arange(0.1, 0.91, 0.01):
    _pred = (y_pred_proba > t).astype(int)
    _f1   = f1_score(y_val, _pred, zero_division=0)
    if _f1 > best_f1:
        best_f1, best_thresh = _f1, t
print(f"\nEnsemble best threshold : {best_thresh:.2f}  →  val-F1={best_f1:.4f}")

y_pred = (y_pred_proba > best_thresh).astype(int)

acc       = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall    = recall_score(y_val, y_pred, zero_division=0)
f1        = f1_score(y_val, y_pred, zero_division=0)

print(f"\n--- Ensemble validation metrics at threshold={best_thresh:.2f} ---")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
report = classification_report(y_val, y_pred, target_names=['Not Taken (0)', 'Taken (1)'])
print(report)

print("Confusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(cm)

# ── Save best model + ensemble weights ───────────────────────────────────────
model_path = os.path.join(OUTPUT_DIR, f'activity_model_{timestamp}.h5')
model.save(model_path)
print(f"\nBest model saved : {model_path}")

# Save all ensemble model paths
for i, m in enumerate(all_models):
    ep = os.path.join(OUTPUT_DIR, f'activity_model_{timestamp}_run{i+1}.h5')
    m.save(ep)

# Save ensemble config (weights + threshold) for inference
import json
ensemble_cfg = {
    'timestamp': timestamp,
    'model_files': [f'activity_model_{timestamp}_run{i+1}.h5' for i in range(len(all_models))],
    'ensemble_weights': weights.tolist(),
    'best_threshold': float(best_thresh),
    'train_columns': list(X_encoded.columns),
}
ensemble_path = os.path.join(OUTPUT_DIR, f'ensemble_config_{timestamp}.json')
with open(ensemble_path, 'w') as f:
    json.dump(ensemble_cfg, f, indent=2)
print(f"Ensemble config : {ensemble_path}")

metrics_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Timestamp     : {timestamp}\n")
    f.write(f"Train data    : act/Data/train.csv\n")
    f.write(f"Train samples : {X_scaled.shape[0]}\n")
    f.write(f"Val   samples : {X_val_scaled.shape[0]}\n")
    f.write(f"Features      : {X_scaled.shape[1]}\n") 
    f.write(f"N restarts    : {N_RESTARTS}\n")
    f.write(f"Ensemble weights: {[f'{w:.3f}' for w in weights]}\n\n")
    f.write(f"--- Best single model (threshold=0.5) ---\n")
    f.write(f"Accuracy   : {acc_keras:.4f}\n")
    f.write(f"Precision  : {prec_keras:.4f}\n")
    f.write(f"Recall     : {rec_keras:.4f}\n")
    f.write(f"F1 Score   : {f1_keras:.4f}\n\n")
    f.write(f"--- Ensemble threshold={best_thresh:.2f} (auto-optimised) ---\n")
    f.write(f"Accuracy   : {acc:.4f}\n")
    f.write(f"Precision  : {precision:.4f}\n")
    f.write(f"Recall     : {recall:.4f}\n")
    f.write(f"F1 Score   : {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
print(f"Metrics saved: {metrics_path}")
