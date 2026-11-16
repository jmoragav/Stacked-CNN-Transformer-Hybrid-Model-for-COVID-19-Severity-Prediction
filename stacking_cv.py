# ---------------------------------------------------------------
#  Stacked ensemble, 5-fold cross-validation
#     python stacking_cv.py             
# Needs storage/ from prepare_data.py. Train, validation and test arrays are merged into one
# dataset, then split into 5 stratified folds.
# Writes storage/cv/cv_fold<k>.npz and storage/cv/cv_per_fold.csv.


import os
import numpy as np, pandas as pd, tensorflow as tf
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from common import STORAGE, load_splits, build_fusion_model, build_simple_cnn

N_FOLDS = 5
CV_DIR = f"{STORAGE}/cv"
os.makedirs(CV_DIR, exist_ok=True)


def fit_fusion(inputs_tr, y_tr, inputs_va, y_va, **flags):
    
    tf.keras.backend.clear_session()
    model = build_fusion_model(**flags)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(inputs_tr, y_tr, validation_data=(inputs_va, y_va), epochs=120, batch_size=4, verbose=2,
              callbacks=[EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss")])
    return model


def metrics(y, p):
    pred = (p > 0.5).astype(int)
    return {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred, average="macro"), "auc": roc_auc_score(y, p)}


# ---- merge the 80/10/10 split into one dataset (order: train, val, test) 
(Xs_tr, Xs_va, Xs_te, E_tr, E_va, E_te, M_tr, M_va, M_te, y_tr, y_va, y_te) = load_splits()
X_seq = np.concatenate([Xs_tr, Xs_va, Xs_te]).astype("float32")
ESM = np.concatenate([E_tr, E_va, E_te]).astype("float32")
META = np.concatenate([M_tr, M_va, M_te]).astype("float32")
y = np.concatenate([y_tr, y_va, y_te]).astype(int)
del Xs_tr, Xs_va, Xs_te, E_tr, E_va, E_te, M_tr, M_va, M_te
print("merged:", X_seq.shape, ESM.shape, META.shape, "Mild share", y.mean().round(3))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
for k, (rest_idx, te_idx) in enumerate(skf.split(X_seq, y)):
    out_path = f"{CV_DIR}/cv_fold{k}.npz"
    if os.path.exists(out_path):
        print(f"fold {k} already done, skipping"); continue
    tr_idx, va_idx = train_test_split(rest_idx, test_size=0.125, stratify=y[rest_idx], random_state=k)
    tf.keras.utils.set_random_seed(k)
    print(f"\n===== FOLD {k}: train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)} =====")

    # Expert 3: XGBoost on metadata
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=5, eval_metric="logloss", random_state=0)
    xgb.fit(META[tr_idx], y[tr_idx])
    m2_va, m2_te = xgb.predict_proba(META[va_idx])[:, 1], xgb.predict_proba(META[te_idx])[:, 1]

    # Expert 2: sequence-only 1D-CNN
    tf.keras.backend.clear_session()
    cnn = build_simple_cnn(X_seq.shape[1])
    cnn.fit(X_seq[tr_idx][..., None], y[tr_idx], validation_data=(X_seq[va_idx][..., None], y[va_idx]),
            epochs=50, batch_size=32, verbose=2, callbacks=[EarlyStopping(patience=8, restore_best_weights=True)])
    m3_va = cnn.predict(X_seq[va_idx][..., None], batch_size=32).flatten()
    m3_te = cnn.predict(X_seq[te_idx][..., None], batch_size=32).flatten()

    # Expert 1: hybrid = full fusion (ESM + CNN-Transformer + metadata)
    hyb = fit_fusion([ESM[tr_idx], X_seq[tr_idx], META[tr_idx]], y[tr_idx],
                     [ESM[va_idx], X_seq[va_idx], META[va_idx]], y[va_idx],
                     use_esm=True, use_cnn_transformer=True, use_metadata=True,
                     esm_dim=ESM.shape[1], meta_dim=META.shape[1])
    m1_va = hyb.predict([ESM[va_idx], X_seq[va_idx], META[va_idx]], batch_size=4).flatten()
    m1_te = hyb.predict([ESM[te_idx], X_seq[te_idx], META[te_idx]], batch_size=4).flatten()

    # Extra comparator: ESM + metadata network
    em = fit_fusion([ESM[tr_idx], META[tr_idx]], y[tr_idx], [ESM[va_idx], META[va_idx]], y[va_idx],
                    use_esm=True, use_cnn_transformer=False, use_metadata=True,
                    esm_dim=ESM.shape[1], meta_dim=META.shape[1])
    em_te = em.predict([ESM[te_idx], META[te_idx]], batch_size=4).flatten()

    # Stack: logistic regression fit on the VALIDATION predictions only, evaluated on the fold's test part
    Z_va = np.column_stack([m1_va, m3_va, m2_va]); Z_te = np.column_stack([m1_te, m3_te, m2_te])
    meta = LogisticRegression(penalty="l2", C=1.0, max_iter=1000).fit(Z_va, y[va_idx])
    stack_te = meta.predict_proba(Z_te)[:, 1]

    np.savez(out_path, te_idx=te_idx, y_te=y[te_idx], hybrid=m1_te, cnn=m3_te, xgb=m2_te,
             esm_meta=em_te, stack=stack_te, coef=meta.coef_)
    print({n: {a: round(b, 4) for a, b in metrics(y[te_idx], p).items()}
           for n, p in [("hybrid", m1_te), ("esm_meta", em_te), ("xgb", m2_te), ("cnn", m3_te), ("stack", stack_te)]})
    tf.keras.backend.clear_session()

# ---- summary over folds
rows = []
for k in range(N_FOLDS):
    p = f"{CV_DIR}/cv_fold{k}.npz"
    if not os.path.exists(p): continue
    d = np.load(p)
    for name in ["cnn", "xgb", "hybrid", "esm_meta", "stack"]:
        rows.append({"fold": k, "model": name, **metrics(d["y_te"], d[name])})
df = pd.DataFrame(rows)
df.to_csv(f"{CV_DIR}/cv_per_fold.csv", index=False)
print(df.groupby("model")[["acc", "f1", "auc"]].agg(["mean", "std"]).round(4))
