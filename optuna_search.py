# ---------------------------------------------------------------
# Step 2. Hyperparameter search for the CNN-Transformer sequence branch (Optuna).
#     python optuna_search.py            (runs up to N_TRIALS trials; start again to resume)
# Needs storage/ from prepare_data.py. The branch uses the biophysical matrix + metadata only

import os, gc, time
import numpy as np, pandas as pd, tensorflow as tf, optuna
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout, Dense,
                                     GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from common import STORAGE, TransformerEncoderLayer

N_TRIALS = 50
RESULTS = f"{STORAGE}/optuna_results.csv"
os.makedirs(STORAGE, exist_ok=True)

L = lambda n: np.load(f"{STORAGE}/{n}.npy")
X_seq_tr, X_meta_tr, y_tr = L("X_seq_train"), L("X_meta_train"), L("y_train")
X_seq_va, X_meta_va, y_va = L("X_seq_val"), L("X_meta_val"), L("y_val")
X_seq_te, X_meta_te, y_te = L("X_seq_test"), L("X_meta_test"), L("y_test")
META_DIM = X_meta_tr.shape[1]


def build_model(p):
    seq_in = Input(shape=(X_seq_tr.shape[1], 1))
    x = seq_in
    for f in [128, 64, 64, 24][:p["num_cnn_layers"]]:
        x = Conv1D(f, kernel_size=4, padding="same", activation="relu")(x)
        x = MaxPooling1D(pool_size=4)(x)
        x = Dropout(p["cnn_dropout"])(x)
    if p["num_cnn_layers"] == 2:
        x = MaxPooling1D(pool_size=2)(x)
    x = Dense(p["d_model"])(x)
    x = TransformerEncoderLayer(d_model=p["d_model"], num_heads=p["num_heads"], dff=p["d_model"] * 4,
                                rate=p["transformer_dropout"])(x)
    x = GlobalAveragePooling1D()(x)
    meta_in = Input(shape=(META_DIM,))
    x = Concatenate()([x, Dense(32, activation="relu")(meta_in)])
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, activation="sigmoid", dtype="float32")(x)
    return tf.keras.Model([seq_in, meta_in], out)


def scores(y, pred):
    try:
        auc = roc_auc_score(y, pred)
    except ValueError:
        auc = float("nan")
    return dict(f1=f1_score(y, pred), accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), roc_auc=auc)


def objective(trial):
    t0 = time.time()
    p = dict(num_cnn_layers=trial.suggest_int("num_cnn_layers", 2, 3),
             d_model=trial.suggest_categorical("d_model", [64, 128, 256]),
             num_heads=trial.suggest_int("num_heads", 4, 6),
             cnn_dropout=trial.suggest_float("cnn_dropout", 0.05, 0.1),
             transformer_dropout=trial.suggest_float("transformer_dropout", 0.05, 0.1),
             learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
             batch_size=trial.suggest_int("batch_size", 2, 4))
    tf.keras.backend.clear_session(); gc.collect()
    set_global_policy(Policy("mixed_float16"))
    model = build_model(p)
    model.compile(optimizer=tf.keras.optimizers.Adam(p["learning_rate"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit([X_seq_tr, X_meta_tr], y_tr, epochs=100, batch_size=p["batch_size"],
                     validation_data=([X_seq_va, X_meta_va], y_va), verbose=2,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    pv = (model.predict([X_seq_va, X_meta_va], batch_size=p["batch_size"]) > 0.5).astype(int).ravel()
    pt = (model.predict([X_seq_te, X_meta_te], batch_size=p["batch_size"]) > 0.5).astype(int).ravel()
    sv, st = scores(y_va, pv), scores(y_te, pt)
    row = dict(trial=trial.number, **p, val_loss=hist.history["val_loss"][-1],
               **{f"val_{k}": v for k, v in sv.items()}, **{f"test_{k}": v for k, v in st.items()},
               runtime_hours=(time.time() - t0) / 3600)
    pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not os.path.exists(RESULTS), index=False)
    print(f"trial {trial.number}: val F1 {sv['f1']:.4f}")
    tf.keras.backend.clear_session(); gc.collect()
    return sv["f1"]


if __name__ == "__main__":
    study = optuna.create_study(study_name="cnn_transformer_optimization",
                                storage=f"sqlite:///{STORAGE}/optuna_study.db",
                                direction="maximize", load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done)
    print("best trial:", study.best_trial.number, "val F1", round(study.best_value, 4))
    print("best params:", study.best_params)
