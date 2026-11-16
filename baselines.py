# ---------------------------------------------------------------
# Step 6. Baselines on the same folds as the stack 
# k-NN, SVM, random forest and XGBoost, each on metadata and on ESM-2 + metadata.
#     python baselines.py
# Needs storage/ from prepare_data.py and storage/cv/ from stacking_cv.py (for the fold indices).
# Writes storage/baselines_per_fold.csv.


import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from common import STORAGE, load_splits

CV = f"{STORAGE}/cv"
(_, _, _, E_tr, E_va, E_te, M_tr, M_va, M_te, y_tr, y_va, y_te) = load_splits()
META = np.concatenate([M_tr, M_va, M_te]).astype("float32")
ESM = np.concatenate([E_tr, E_va, E_te]).astype("float32")
y = np.concatenate([y_tr, y_va, y_te]).astype(int)
n = len(y)
folds = [np.load(f"{CV}/cv_fold{k}.npz") for k in range(5)]
assert all((y[f["te_idx"]] == f["y_te"]).all() for f in folds), "fold labels do not match the data order"

feats = {"metadata": META, "esm+metadata": np.hstack([ESM, META])}


def models():
    return {"k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Machine": SVC(probability=True, kernel="rbf", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, reg_lambda=5, eval_metric="logloss", random_state=0)}


rows = []
for k, f in enumerate(folds):
    te = f["te_idx"]; rest = np.setdiff1d(np.arange(n), te)
    tr, va = train_test_split(rest, test_size=0.125, stratify=y[rest], random_state=k)   # same training rows as the stack
    for fname, X in feats.items():
        for name, m in models().items():
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]; pr = (p > 0.5).astype(int)
            rows.append(dict(fold=k, features=fname, model=name, acc=accuracy_score(y[te], pr),
                             f1=f1_score(y[te], pr, average="macro"), auc=roc_auc_score(y[te], p)))
    print("fold", k, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{STORAGE}/baselines_per_fold.csv", index=False)
print(df.groupby(["features", "model"])[["acc", "f1", "auc"]].agg(["mean", "std"]).round(4).to_string())
