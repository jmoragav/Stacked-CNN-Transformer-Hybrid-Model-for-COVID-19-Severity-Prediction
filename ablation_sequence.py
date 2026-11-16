# ---------------------------------------------------------------
# Sequence-branch ablation
# Run as a script from the folder that holds common.py, supplementary_table_s1.ods
# (and, after prepare_data.py, storage/ for the split consistency check):
#     python ablation_sequence.py
# All variants use BEST_PARAMS  and all 128 metadata columns.
# Results are appended to storage/ablation_fixed_results.csv after every variant

import os, gc, time
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout, Dense,
                                     GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from common import *   # BEST_PARAMS, constants, TransformerEncoderLayer, load_data_full, fasta_to_numeric

VARIANTS = [
    dict(name="full",           tr=True,  ncnn=3, rbd=True,  struct=True,  hb=True,  meta=True),
    dict(name="no_transformer", tr=False, ncnn=3, rbd=True,  struct=True,  hb=True,  meta=True),
    dict(name="fewer_cnn",      tr=True,  ncnn=2, rbd=True,  struct=True,  hb=True,  meta=True),
    dict(name="no_rbd_weight",  tr=True,  ncnn=3, rbd=False, struct=True,  hb=True,  meta=True),
    dict(name="no_structure",   tr=True,  ncnn=3, rbd=True,  struct=False, hb=True,  meta=True),
    dict(name="no_hbond",       tr=True,  ncnn=3, rbd=True,  struct=True,  hb=False, meta=True),
    dict(name="no_metadata",    tr=True,  ncnn=3, rbd=True,  struct=True,  hb=True,  meta=False),
]
os.makedirs(STORAGE, exist_ok=True)
OUT_CSV = f"{STORAGE}/ablation_fixed_results.csv"
SEEDS = [0, 1, 2]   # pass 1 gives a full table, later passes add seeds

# ---------- data and split (same random_state as before)
sequences, X_meta, y = load_data_full(DATA_FILE)
y = np.asarray(y)
print("metadata columns:", X_meta.shape[1], "(should be 128)")
assert X_meta.shape[1] == 128, "unexpected number of metadata columns"
N_META = X_meta.shape[1]
idx = np.arange(len(y))
idx_tv, idx_te = train_test_split(idx, test_size=0.1, random_state=42, stratify=y)
idx_tr, idx_va = train_test_split(idx_tv, test_size=0.1111, random_state=42, stratify=y[idx_tv])

# sanity check 
if os.path.exists(f'{STORAGE}/y_test.npy'):
    assert np.array_equal(y[idx_te], np.load(f'{STORAGE}/y_test.npy')), "test labels differ from saved split"
    assert np.array_equal(y[idx_va], np.load(f'{STORAGE}/y_val.npy')), "val labels differ from saved split"
    assert np.array_equal(y[idx_tr], np.load(f'{STORAGE}/y_train.npy')), "train labels differ from saved split"
    _m = np.load(f'{STORAGE}/X_meta_test.npy'); _c = min(_m.shape[1], X_meta.shape[1])
    assert np.allclose(X_meta[idx_te][:, :_c], _m[:, :_c]), "test metadata differs"
    print("Split check passed: labels and metadata match the saved split.")

def build(v):
    seq_in = Input(shape=(16581, 1))
    x = seq_in
    for f in BEST_PARAMS["filters"][:v["ncnn"]]:
        x = Conv1D(f, 4, padding="same", activation="relu")(x)
        x = MaxPooling1D(pool_size=4)(x)
        x = Dropout(BEST_PARAMS["cnn_dropout"])(x)
    if v["ncnn"] == 2:
        x = MaxPooling1D(pool_size=2)(x)
    x = Dense(BEST_PARAMS["d_model"])(x)
    if v["tr"]:
        x = TransformerEncoderLayer(d_model=BEST_PARAMS["d_model"], num_heads=BEST_PARAMS["heads"],
                                    dff=BEST_PARAMS["d_model"] * 4, rate=BEST_PARAMS["tr_dropout"])(x)
    x = GlobalAveragePooling1D()(x)
    if v["meta"]:
        m_in = Input(shape=(N_META,))
        x = Concatenate()([x, Dense(32, activation="relu")(m_in)])
        inputs = [seq_in, m_in]
    else:
        inputs = [seq_in]
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, activation="sigmoid", dtype="float32")(x)
    return tf.keras.Model(inputs, out)

done = set()
if os.path.exists(OUT_CSV):
    _d = pd.read_csv(OUT_CSV)
    done = set(zip(_d["variant"], _d["seed"]))
set_global_policy(Policy("mixed_float16"))
feat_cache = {}   # sequence features per feature setting (about 200 MB each)

for seed in SEEDS:
    for v in VARIANTS:
        if (v["name"], seed) in done:
            print("skip", v["name"], seed); continue
        t0 = time.time()
        key = (v["rbd"], v["struct"], v["hb"])
        if key not in feat_cache:
            cfg = dict(rbd_weight=v["rbd"], use_structure=v["struct"], use_hbond=v["hb"])
            feat_cache[key] = np.array([fasta_to_numeric(s_, cfg) for s_ in sequences], dtype=np.float32)[..., None]
        X = feat_cache[key]
        def inp(ix):
            return [X[ix], X_meta[ix]] if v["meta"] else [X[ix]]

        tf.keras.backend.clear_session(); gc.collect()
        tf.keras.utils.set_random_seed(seed)
        model = build(v)
        model.compile(optimizer=tf.keras.optimizers.Adam(BEST_PARAMS["lr"]), loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(inp(idx_tr), y[idx_tr], epochs=100, batch_size=BEST_PARAMS["batch"],
                  validation_data=(inp(idx_va), y[idx_va]),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
                  verbose=2)

        prob = model.predict(inp(idx_te), batch_size=BEST_PARAMS["batch"]).ravel().astype(float)
        pred = (prob > 0.5).astype(int)
        np.save(f"{STORAGE}/probs_fixed_{v['name']}_seed{seed}.npy", prob)
        row = dict(variant=v["name"], seed=seed,
                   accuracy=accuracy_score(y[idx_te], pred),
                   macro_f1=f1_score(y[idx_te], pred, average="macro"),
                   mild_f1=f1_score(y[idx_te], pred),
                   roc_auc=roc_auc_score(y[idx_te], prob),
                   hours=(time.time() - t0) / 3600)
        pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)
        print(row)
        del model; gc.collect()

res = pd.read_csv(OUT_CSV)
print(res.groupby("variant")[["accuracy", "macro_f1", "roc_auc"]].agg(["mean", "std"]).round(4).to_string())
