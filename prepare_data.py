# ---------------------------------------------------------------
# Step 1. Data preparation: filter the table, split 80/10/10, build the biophysical
# matrix and the ESM-2 embeddings, and save everything to storage/.
#     python prepare_data.py
# Needs: supplementary_table_s1.ods (this folder), torch + transformers (ESM-2, a GPU helps).

# ---------------------------------------------------------------
import os, gc
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from common import STORAGE, DATA_FILE, load_data_full, fasta_to_numeric

os.makedirs(STORAGE, exist_ok=True)
SEQ_CONFIG = dict(rbd_weight=True, use_structure=True, use_hbond=True)   # features used for every model
ESM_NAME, ESM_MAX_LEN, ESM_BATCH = "facebook/esm2_t30_150M_UR50D", 512, 2

# ---- filter, labels (Mild = 1, Severe = 0) and metadata 
sequences, X_meta, y = load_data_full(DATA_FILE)
y = np.asarray(y, dtype=np.int32)
print(f"{len(y)} samples | Mild {int(y.sum())} | Severe {int((1 - y).sum())} | metadata columns {X_meta.shape[1]}")

# ---- 80/10/10 split, stratified
idx = np.arange(len(y))
idx_tv, idx_te = train_test_split(idx, test_size=0.1, random_state=42, stratify=y)
idx_tr, idx_va = train_test_split(idx_tv, test_size=0.1111, random_state=42, stratify=y[idx_tv])
splits = {"train": idx_tr, "val": idx_va, "test": idx_te}

# ---- biophysical matrix 
X_seq = np.array([fasta_to_numeric(s, SEQ_CONFIG) for s in sequences], dtype=np.float32)

# ---- ESM-2 embeddings
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(ESM_NAME)
esm = AutoModel.from_pretrained(ESM_NAME, torch_dtype=dtype).to(device).eval()


def embed(seqs):
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), ESM_BATCH), desc="ESM-2"):
            batch = [s[:ESM_MAX_LEN] for s in seqs[i:i + ESM_BATCH]]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=ESM_MAX_LEN)
            enc = {k: v.to(device) for k, v in enc.items()}
            h = esm(**enc).last_hidden_state
            out.append(h.float().mean(dim=1).cpu().numpy())
            del h, enc
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    return np.vstack(out)


# ---- save ----
for name, ix in splits.items():
    np.save(f"{STORAGE}/y_{name}.npy", y[ix])
    np.save(f"{STORAGE}/X_meta_{name}.npy", X_meta[ix])
    np.save(f"{STORAGE}/X_seq_{name}.npy", X_seq[ix])
    np.save(f"{STORAGE}/esm_{name}.npy", embed([sequences[i] for i in ix]))
    print("saved", name, len(ix))

order = np.concatenate([idx_tr, idx_va, idx_te])
ids, _ = pd.factorize(pd.Series([sequences[i] for i in order]))
np.save(f"{STORAGE}/seq_ids.npy", ids.astype(np.int64))
print("unique sequences:", ids.max() + 1)
