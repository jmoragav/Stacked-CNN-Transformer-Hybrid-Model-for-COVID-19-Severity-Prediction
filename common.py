"""Shared helpers for the ablation scripts: constants, TransformerEncoderLayer,
data loading and sequence-to-numeric conversion.
Used with:  from common import *
"""
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense

logger = logging.getLogger(__name__)

# Best Optuna trial paper
BEST_PARAMS = dict(filters=[128, 64, 64], cnn_dropout=0.10539767112071488, d_model=64, heads=5,
                   tr_dropout=0.06694009281622051, lr=0.0002056743841367156, batch=3)

# Spike RBD position range and amino-acid property tables
RBD_START = 319
RBD_END = 542
sec_struc = {'H': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}
aa_props = {
    'A': [0.62, -0.5, 1.8, 0, 0, 1, 1, 0, 0, 0], 'C': [0.29, -1, 2.5, 0, 0, 1, 0, 1, 0, 0],
    'D': [-0.9, 3, -3.5, 0, 1, 0, 0, 0, 0, 0], 'E': [-0.74, 3, -3.5, 0, 1, 0, 0, 0, 0, 0],
    'F': [1.19, -2.5, 2.8, 0, 0, 1, 1, 0, 0, 0], 'G': [0.48, 0, -0.4, 1, 0, 0, 0, 0, 0, 1],
    'H': [-0.4, -0.5, -3.2, 0, 0, 1, 0, 0, 1, 0], 'I': [1.38, -1.8, 4.5, 0, 0, 1, 1, 0, 0, 0],
    'K': [-1.5, 3, -3.9, 0, 1, 0, 0, 0, 0, 0], 'L': [1.06, -1.8, 3.8, 0, 0, 1, 1, 0, 0, 0],
    'M': [0.64, -1.3, 1.9, 0, 0, 1, 1, 1, 0, 0], 'N': [-0.78, 2, -3.5, 0, 1, 0, 0, 0, 0, 0],
    'P': [0.12, 0, -1.6, 0, 0, 0, 0, 0, 0, 0], 'Q': [-0.85, 2, -3.5, 0, 1, 0, 0, 0, 1, 0],
    'R': [-2.53, 3, -4.5, 0, 1, 0, 0, 0, 1, 0], 'S': [-0.18, 0.3, -0.8, 1, 0, 0, 0, 0, 0, 1],
    'T': [-0.05, -0.4, -0.7, 1, 0, 1, 0, 0, 0, 1], 'V': [1.08, -1.5, 4.2, 0, 0, 1, 1, 0, 0, 0],
    'W': [0.81, -3.4, -0.9, 0, 0, 1, 1, 0, 0, 0], 'Y': [0.26, -2.3, -1.3, 0, 0, 1, 1, 0, 0, 0]
}



class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def remove_inconclusive(df):
    logger.info("Removing inconclusive patient statuses")
    inconclusive_tags = ['ALIVE', 'Alive, hospitalized', 'Emergency Care', 'Hospitalized', 'Inpatient', 'Live', 'moderate symptomatic, hospita', 'Moderate']
    inconclusive_tags = [tag.lower() for tag in inconclusive_tags]
    df_filtered = df[~df['Patient status'].str.lower().isin(inconclusive_tags)].copy()
    logger.info(f"Removed {len(df) - len(df_filtered)} inconclusive rows. Filtered shape: {df_filtered.shape}")
    return df_filtered

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_excel(file_path, engine='odf')
    logger.info(f"Data loaded. Initial shape: {df.shape}")
    df = remove_inconclusive(df)
    df = df.rename(columns={"Patient age": "Age"})
    df['status'] = df['status'].str.lower().apply(
        lambda x: 1 if x == "mild" else (0 if x == "severe" else x)
    )
    invalid_status = ~df['status'].isin([0, 1])
    if invalid_status.any():
        logger.warning(f"Found {invalid_status.sum()} invalid status values at indices: {invalid_status[invalid_status].index.tolist()}")
        df = df[~invalid_status].copy()
    df['status'] = df['status'].astype(np.int32)
    logger.info("Labels mapped (mild=1, severe=0) and converted to int32")
    
    expected_columns = ['Fastas', 'Gender', 'Age', 'Clade', 'Lineage', 'status']
    if not all(col in df.columns for col in expected_columns):
        logger.error(f"Expected columns: {expected_columns}")
        logger.error(f"Found columns: {list(df.columns)}")
        raise ValueError("Missing or incorrect column names in ODS file")
    
    logger.info("Extracting FASTA sequences")
    sequences = df['Fastas'].str.replace('\n', '').tolist()
    logger.info(f"Extracted {len(sequences)} sequences")
    
    logger.info("Processing metadata")
    metadata = df[['Gender', 'Age', 'Clade', 'Lineage']].copy()
    metadata['Age'] = metadata['Age'] / 100.0
    metadata = pd.get_dummies(metadata, columns=['Gender', 'Clade', 'Lineage'], dtype=np.float32)
    if metadata.shape[1] > 41:
        logger.warning(f"Metadata has {metadata.shape[1]} features, truncating to 41")
        metadata = metadata.iloc[:, :41]
    elif metadata.shape[1] < 41:
        logger.warning(f"Metadata has {metadata.shape[1]} features, padding to 41")
        metadata = pd.concat([metadata, pd.DataFrame(0.0, index=metadata.index, columns=[f'pad_{i}' for i in range(41 - metadata.shape[1])])], axis=1)
    logger.info(f"Metadata processed. Shape: {metadata.shape}")
    
    labels = df['status'].values
    logger.info(f"Labels extracted. Shape: {labels.shape}")
    logger.info(f"Class distribution: mild={np.sum(labels == 1)}, severe={np.sum(labels == 0)}")
    
    return sequences, metadata.values.astype(np.float32), labels.astype(np.int32)

def fasta_to_numeric(seq, config):
    FIXED_SIZE = 16581
    if seq.startswith('>'):
        seq = seq.split('\n', 1)[1].replace('\n', '')
    else:
        seq = seq.replace('\n', '')
    amino_acids = list(seq)
    numeric_seq = []
    hbond_aa_freq = []
    
    if config['use_hbond']:
        for aa in ['S', 'T', 'N', 'Q', 'H', 'Y']:
            hbond_aa_freq.append(amino_acids.count(aa) / len(amino_acids))
    
    for i, aa in enumerate(amino_acids):
        penalty = 5 if config['rbd_weight'] and RBD_START <= i <= RBD_END else 1
        numeric_seq.append([prop * penalty for prop in aa_props.get(aa, [0] * 10)])
        if config['use_structure']:
            sec_struc_aa = 'H' if RBD_START <= i <= RBD_END else 'C'
            numeric_seq.append(sec_struc.get(sec_struc_aa, [0, 0, 0]))
    
    if config['use_hbond']:
        numeric_seq.append(hbond_aa_freq)
    
    numeric_seq = np.concatenate(numeric_seq)
    if len(numeric_seq) < FIXED_SIZE:
        numeric_seq = np.pad(numeric_seq, (0, FIXED_SIZE - len(numeric_seq)), 'constant')
    elif len(numeric_seq) > FIXED_SIZE:
        numeric_seq = numeric_seq[:FIXED_SIZE]
    
    return numeric_seq.astype(np.float32)


def load_data_full(path):
    """Same rows and order as load_data, but keeps all one-hot metadata columns (no cut to 41)."""
    df = pd.read_excel(path, engine='odf')
    df = remove_inconclusive(df)
    df = df.rename(columns={"Patient age": "Age"})
    df['status'] = df['status'].str.lower().apply(lambda x: 1 if x == "mild" else (0 if x == "severe" else x))
    df = df[df['status'].isin([0, 1])].copy()
    yy = df['status'].astype(np.int32).values
    seqs = df['Fastas'].str.replace('\n', '').tolist()
    meta = df[['Gender', 'Age', 'Clade', 'Lineage']].copy()
    meta['Age'] = meta['Age'] / 100.0
    meta = pd.get_dummies(meta, columns=['Gender', 'Clade', 'Lineage'], dtype=np.float32)
    return seqs, meta.values.astype(np.float32), yy


# ----------------------------------------------------------------------------------
# Paths and shared model code used by the training scripts

import os
STORAGE = "storage"          # arrays and results are read / written to this folder
DATA_FILE = "supplementary_table_s1.ods"


def load_splits(storage=STORAGE):
    """Arrays written by prepare_data.py (the same 80/10/10 split everywhere)."""
    L = lambda n: np.load(f"{storage}/{n}.npy")
    return (L("X_seq_train"), L("X_seq_val"), L("X_seq_test"),
            L("esm_train"), L("esm_val"), L("esm_test"),
            L("X_meta_train"), L("X_meta_val"), L("X_meta_test"),
            L("y_train"), L("y_val"), L("y_test"))


# Settings of Optuna trial used by the hybrid network
FUSION_PARAMS = dict(filters=[128, 64, 64], cnn_dropout=0.105, d_model=64, heads=5, tr_dropout=0.067)


def build_fusion_model(use_esm=True, use_cnn_transformer=True, use_metadata=True, esm_dim=0, meta_dim=128):
    """Hybrid network: optional ESM-2 embedding + CNN-Transformer on the biophysical matrix + metadata."""
    from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout, Dense,
                                         GlobalAveragePooling1D, Concatenate)
    P = FUSION_PARAMS
    inputs, features = [], []
    if use_esm:
        esm_in = Input(shape=(esm_dim,), name="esm_input")
        inputs.append(esm_in); features.append(esm_in)
    if use_cnn_transformer:
        cnn_in = Input(shape=(16581, 1), name="cnn_input")
        x = cnn_in
        for f in P["filters"]:
            x = Conv1D(f, kernel_size=4, padding="same", activation="relu")(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(P["cnn_dropout"])(x)
        x = Dense(P["d_model"])(x)
        x = TransformerEncoderLayer(d_model=P["d_model"], num_heads=P["heads"],
                                    dff=P["d_model"] * 4, rate=P["tr_dropout"])(x)
        x = GlobalAveragePooling1D()(x)
        inputs.append(cnn_in); features.append(x)
    if use_metadata:
        meta_in = Input(shape=(meta_dim,), name="meta_input")
        inputs.append(meta_in); features.append(Dense(32, activation="relu")(meta_in))
    x = Concatenate()(features) if len(features) > 1 else features[0]
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=out)


def build_simple_cnn(n_feat):
    """Sequence-only 1D-CNN expert of the stacked ensemble."""
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_feat, 1)),
        tf.keras.layers.Conv1D(32, 7, activation="relu"),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(64, 5, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m
