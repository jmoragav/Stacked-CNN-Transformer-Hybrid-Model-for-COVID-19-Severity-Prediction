# ---------------------------------------------------------------
# Step 4. Ablation of the hybrid model (Table 9, Figures 6-8):
# full fusion / no ESM / ESM + metadata / ESM only, five seeds each.
#     python ablation_hybrid.py          (resumable: finished runs are skipped)
# Needs storage/ from prepare_data.py.
# Writes storage/probs_<config>_seed<k>.npy, history_<config>_seed<k>.pkl,
# done_<config>_seed<k>.json and storage/ablation_all_seeds.csv.
# ---------------------------------------------------------------
import os, json, glob, pickle, time
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from common import STORAGE, load_splits, build_fusion_model

CONFIGS = [
    dict(name="full_fusion", use_esm=True,  use_cnn_transformer=True,  use_metadata=True),
    dict(name="no_esm",      use_esm=False, use_cnn_transformer=True,  use_metadata=True),
    dict(name="esm_metada",  use_esm=True,  use_cnn_transformer=False, use_metadata=True),
    dict(name="esm_only",    use_esm=True,  use_cnn_transformer=False, use_metadata=False),
]
SEEDS = range(5)
os.makedirs(STORAGE, exist_ok=True)

(Xs_tr, Xs_va, Xs_te, E_tr, E_va, E_te, M_tr, M_va, M_te, y_tr, y_va, y_te) = load_splits()


def inputs(cfg, esm, seq, meta):
    xs = []
    if cfg["use_esm"]: xs.append(esm)
    if cfg["use_cnn_transformer"]: xs.append(seq)
    if cfg["use_metadata"]: xs.append(meta)
    return xs


for seed in SEEDS:
    for cfg in CONFIGS:
        name = cfg["name"]
        done = f"{STORAGE}/done_{name}_seed{seed}.json"
        if os.path.exists(done):
            print("skip", name, seed); continue
        t0 = time.time()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_fusion_model(cfg["use_esm"], cfg["use_cnn_transformer"], cfg["use_metadata"],
                                   esm_dim=E_tr.shape[1], meta_dim=M_tr.shape[1])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
        h = model.fit(inputs(cfg, E_tr, Xs_tr, M_tr), y_tr,
                      validation_data=(inputs(cfg, E_va, Xs_va, M_va), y_va),
                      epochs=120, batch_size=4, verbose=2,
                      callbacks=[EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss")])
        with open(f"{STORAGE}/history_{name}_seed{seed}.pkl", "wb") as f:
            pickle.dump(h.history, f)
        pv = model.predict(inputs(cfg, E_va, Xs_va, M_va)).ravel()
        pt = model.predict(inputs(cfg, E_te, Xs_te, M_te)).ravel()
        np.save(f"{STORAGE}/probs_{name}_seed{seed}.npy", pt)
        m = dict(val_loss=h.history["val_loss"][-1],
                 val_accuracy=accuracy_score(y_va, pv > 0.5), val_f1=f1_score(y_va, pv > 0.5, average="macro"),
                 val_roc_auc=roc_auc_score(y_va, pv),
                 test_f1=f1_score(y_te, pt > 0.5, average="macro"),
                 test_accuracy=accuracy_score(y_te, pt > 0.5), test_roc_auc=roc_auc_score(y_te, pt))
        with open(done, "w") as f:
            json.dump(dict(seed=seed, config=name, runtime=(time.time() - t0) / 3600, **m), f)
        print(name, seed, {k: round(v, 4) for k, v in m.items()})

rows = [json.load(open(p)) for p in glob.glob(f"{STORAGE}/done_*_seed*.json")]
df = pd.DataFrame(rows)
df.to_csv(f"{STORAGE}/ablation_all_seeds.csv", index=False)
print(df.groupby("config")[["test_accuracy", "test_f1", "test_roc_auc"]].agg(["mean", "std"]).round(4))
