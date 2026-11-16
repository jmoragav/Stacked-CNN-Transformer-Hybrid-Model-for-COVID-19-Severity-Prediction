"""Builds every figure and the result tables of the manuscript in one run.

Usage:   python make_all_figures.py [project_folder]        (default: current folder)

Inputs :
    storage/cv/cv_fold0..4.npz, storage/cv/cv_per_fold.csv       cross-validation outputs
    storage/probs_<config>_seed<k>.npy, history_<config>_seed<k>.pkl, ablation_all_seeds.csv
    baselines_per_fold.csv         (optional, Table 5; also looked for inside storage/)
    optuna_results.csv             (optional, Table 8; also looked for inside storage/)
    ablation_fixed_results.csv     (optional, Table 7; also looked for inside storage/)
    seq_ids.npy                    (optional, Table 6)

Outputs:
    figures/Figure_1.png ... Figure_8.png     the figures with the numbers used in the manuscript
    figures/panels/                           single-panel versions of Figure 6 and Figure 8
    tables/Table_4.csv, 5, 6, 7, 8, 9           mean +/- SD tables, ready to paste
"""
import os, sys, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import (confusion_matrix, roc_curve, accuracy_score, f1_score, roc_auc_score)

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
S = os.path.join(ROOT, "storage")
CV = os.path.join(S, "cv")
FIG = os.path.join(ROOT, "figures")
PAN = os.path.join(FIG, "panels")
TAB = os.path.join(ROOT, "tables")
for d in (FIG, PAN, TAB):
    os.makedirs(d, exist_ok=True)

INK, INK2, GRID = "#1e1e1e", "#52514e", "#dcdcd8"
C_HYB, C_STACK, C_XGB, C_CNN = "#2a78d6", "#eb6834", "#1baf7a", "#7a7a7a"
C_TR, C_VA = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#8a8a86", "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
CMAP = LinearSegmentedColormap.from_list("blu", ["#eef4fb", "#2a78d6"])

ABL = [("full_fusion", "a", "Full fusion"), ("no_esm", "b", "No ESM"),
       ("esm_only", "c", "ESM only"), ("esm_metada", "d", "ESM + metadata")]


def find(name):
    for p in (os.path.join(ROOT, name), os.path.join(S, name)):
        if os.path.exists(p):
            return p
    return None


def save(fig, name, **kw):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=300, facecolor="white", **kw)
    plt.close(fig)
    print("saved", path)


# ===================================================================
# Figures 1 and 2: diagrams

BLUE_F, BLUE_E = "#DCE9F7", "#3A6EA5"
GREEN_F, GREEN_E = "#DDF0DD", "#4C8C4A"
ORNG_F, ORNG_E = "#FCE8CF", "#C98A2B"
RED_F, RED_E = "#F8DADA", "#B04A4A"
GRAY_F = "#F5F5F5"


def _tools(ax):
    def box(x, y, w, h, text, fc, ec, fs=11, bold=False, rot=0, lw=1.5, r=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", rotation=rot, zorder=3, linespacing=1.25)

    def arrow(x0, y0, x1, y1, color="#444444", lw=1.6):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                     color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=4))
    return box, arrow


def figure1():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 120); ax.set_ylim(0, 70); ax.axis("off")
    box, arrow = _tools(ax)

    def lane(x, y, w, h, title, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=GRAY_F, ec=ec, lw=1.2, ls="--", zorder=1))
        ax.text(x + 1.2, y + h - 1.6, title, ha="left", va="top", fontsize=11.5,
                fontweight="bold", color=ec, zorder=3)

    box(1, 36, 17, 26, "Spike protein\nsequence\n\n(amino acids,\n1,267–1,275\nresidues)", BLUE_F, BLUE_E)
    box(1, 5, 17, 24, "Clinical\nmetadata\n\nAge, Gender,\nClade, Lineage\n(128 features)", GREEN_F, GREEN_E)
    lane(23, 56, 56, 12, "Expert: sequence-only 1D-CNN", BLUE_E)
    box(25, 57, 52, 6, "Biophysical matrix (16,581 values)   →   1D-CNN   →   probability", BLUE_F, BLUE_E)
    lane(23, 19, 56, 34.5, "Expert: hybrid model (ESM-2 + CNN-Transformer + metadata)", BLUE_E)
    box(25, 43, 21, 6, "Frozen ESM-2 (150M)\n640-d embedding", BLUE_F, BLUE_E, fs=10.5)
    box(25, 32.5, 21, 9, "Biophysical matrix\n1D-CNN + Transformer\n(64-d output)", BLUE_F, BLUE_E, fs=10.5)
    box(25, 22.5, 21, 6, "Metadata\n→ Dense 32", GREEN_F, GREEN_E, fs=10.5)
    box(49, 24, 11, 24, "Concat\n\n736-d", ORNG_F, ORNG_E, fs=10.5)
    box(63, 31, 14, 10, "Dense\n64 → 32 → 16\n→ sigmoid", ORNG_F, ORNG_E, fs=10.5)
    for a in [(46, 46, 49, 46), (46, 37, 49, 37), (46, 25.5, 49, 25.5), (60, 36, 63, 36)]:
        arrow(*a)
    lane(23, 2, 56, 12, "Expert: metadata XGBoost", GREEN_E)
    box(25, 3, 52, 6, "Metadata (128 features)   →   XGBoost   →   probability", GREEN_F, GREEN_E)
    for a in [(18, 60, 25, 60), (18, 46, 25, 46), (18, 37, 25, 37), (18, 25.5, 25, 25.5), (18, 6.0, 25, 6.0)]:
        arrow(*a)
    box(84, 4, 6, 60, "Probability fusion matrix", ORNG_F, ORNG_E, rot=90)
    for a in [(77, 60, 84, 60), (77, 36, 84, 36), (77, 6, 84, 6)]:
        arrow(*a)
    ax.text(80.5, 61.2, "p$_{CNN}$", ha="center", va="bottom", fontsize=9.5, color="#444444")
    ax.text(80.5, 37.2, "p$_{hybrid}$", ha="center", va="bottom", fontsize=9.5, color="#444444")
    ax.text(80.5, 7.2, "p$_{XGB}$", ha="center", va="bottom", fontsize=9.5, color="#444444")
    box(95, 36, 24, 18, "Logistic-regression\nmeta-learner\n(L2 regularized)\n\nfit on validation\npredictions",
        ORNG_F, ORNG_E, fs=10.5)
    box(95, 12, 24, 14, "Severity prediction\n\nMild / Severe", RED_F, RED_E)
    arrow(90, 45, 95, 45); arrow(107, 36, 107, 26)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save(fig, "Figure_1.png")


def figure2():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 110); ax.set_ylim(0, 68); ax.axis("off")
    box, arrow = _tools(ax)

    def title(x, y, text, color):
        ax.text(x, y, text, ha="left", va="bottom", fontsize=11.5, fontweight="bold", color=color)

    def tag(x, y, text):
        ax.text(x, y, text, ha="center", va="bottom", fontsize=9.5, color="#444444", style="italic")

    title(1, 62.3, "Path 1: frozen ESM-2 embedding", BLUE_E)
    box(1, 51, 15, 10, "Spike sequence\n(first 510\nresidues)", BLUE_F, BLUE_E, fs=10.5)
    box(20, 51, 17, 10, "Frozen ESM-2\n(150M, 30 layers)", BLUE_F, BLUE_E, fs=10.5)
    box(41, 51, 17, 10, "Mean pooling\nover tokens", BLUE_F, BLUE_E, fs=10.5)
    arrow(16, 56, 20, 56); arrow(37, 56, 41, 56); arrow(58, 56, 82, 56); tag(70, 56.6, "640-d")

    title(1, 44.8, "Path 2: biophysical CNN-Transformer", BLUE_E)
    box(1, 24, 15, 19, "Spike sequence\n→ biophysical\nmatrix\n(16,581 values)", BLUE_F, BLUE_E, fs=10.5)
    box(20, 24, 17, 19, "3 × Conv1D block\n128, 64, 64 filters\nkernel 4\nmax-pool 2\ndropout 0.105", BLUE_F, BLUE_E, fs=10)
    box(41, 24, 9, 19, "Dense\n64", BLUE_F, BLUE_E, fs=10.5)
    box(54, 24, 15, 19, "Transformer\nencoder\n5 heads\nd = 64\nFFN 256", BLUE_F, BLUE_E, fs=10)
    box(73, 24, 7, 19, "Avg\npool", BLUE_F, BLUE_E, fs=10)
    for a in [(16, 33.5, 20, 33.5), (37, 33.5, 41, 33.5), (50, 33.5, 54, 33.5), (69, 33.5, 73, 33.5), (80, 33.5, 82, 33.5)]:
        arrow(*a)
    tag(77, 43.8, "64-d")

    title(1, 14.8, "Path 3: clinical metadata", GREEN_E)
    box(1, 3.5, 15, 10, "Metadata\n(128 features)", GREEN_F, GREEN_E, fs=10.5)
    box(20, 3.5, 17, 10, "Dense 32\n(ReLU)", GREEN_F, GREEN_E, fs=10.5)
    arrow(16, 8.5, 20, 8.5); arrow(37, 8.5, 82, 8.5); tag(60, 9.1, "32-d")

    box(82, 4, 9, 57, "Concatenate\n640 + 64 + 32 = 736-d", ORNG_F, ORNG_E, fs=10.5, rot=90)
    box(95, 47, 14, 11, "Dense 64\nReLU\ndropout 0.1", ORNG_F, ORNG_E, fs=10)
    box(95, 33, 14, 8, "Dense 32\nReLU", ORNG_F, ORNG_E, fs=10)
    box(95, 19, 14, 8, "Dense 16\nReLU", ORNG_F, ORNG_E, fs=10)
    box(95, 3.5, 14, 10, "Sigmoid output\nP(Mild)", RED_F, RED_E, fs=10)
    arrow(91, 52.5, 95, 52.5); arrow(102, 47, 102, 41); arrow(102, 33, 102, 27); arrow(102, 19, 102, 13.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save(fig, "Figure_2.png")


# ===================================================================
# Figures 3, 4, 5: cross-validated stack

def figures345():
    folds = [np.load(f"{CV}/cv_fold{k}.npz") for k in range(5)]
    y = np.concatenate([f["y_te"] for f in folds])
    P = {m: np.concatenate([f[m] for f in folds]) for m in ["cnn", "xgb", "hybrid", "stack"]}
    auc_mean = pd.read_csv(f"{CV}/cv_per_fold.csv").groupby("model")["auc"].mean()

    # Figure 3: pooled confusion matrix of the stack
    cm = confusion_matrix(y, (P["stack"] > 0.5).astype(int), labels=[0, 1])
    rp = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(rp, cmap=CMAP, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{rp[i, j] * 100:.1f}%", ha="center", va="center", fontsize=13,
                    color="white" if rp[i, j] > 0.55 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Severe", "Mild"]); ax.set_yticklabels(["Severe", "Mild"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); save(fig, "Figure_3.png")

    # Figure 4: pooled ROC curves
    spec = [("stack", "Stacked ensemble", C_STACK, "-", 2.8), ("hybrid", "Hybrid model", C_HYB, "--", 2.0),
            ("xgb", "XGBoost (metadata)", C_XGB, "-.", 2.0), ("cnn", "1D-CNN (sequence only)", C_CNN, ":", 2.0)]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for key, label, col, ls, lw in spec:
        fpr, tpr, _ = roc_curve(y, P[key])
        ax.plot(fpr, tpr, color=col, ls=ls, lw=lw, label=f"{label}  (AUC {auc_mean[key]:.3f})")
    ax.plot([0, 1], [0, 1], color="#b5b5b0", lw=1.2, ls=(0, (1, 3)), label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout(); save(fig, "Figure_4.png")

    # Figure 5: meta-learner coefficients (mean +/- SD over folds)
    co = np.vstack([f["coef"].ravel() for f in folds])     # columns: hybrid, cnn, xgb
    mean, sd = co.mean(0), co.std(0, ddof=1)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    x = np.arange(3)
    ax.bar(x, mean, width=0.55, color=[C_HYB, C_CNN, C_XGB], zorder=2)
    ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=3)
    for i in range(3):
        ax.text(i, mean[i] + sd[i] + 0.12, f"{mean[i]:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(["Hybrid", "1D-CNN", "XGBoost"])
    ax.set_ylabel("Meta-learner coefficient"); ax.set_ylim(0, 3.6)
    ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "Figure_5.png")
    return folds


# ===================================================================
# Figures 6, 7, 8: ablation of the hybrid model

def ablation_data():
    y = np.full(2960, -1)
    for k in range(5):
        f = np.load(f"{CV}/cv_fold{k}.npz")
        y[f["te_idx"]] = f["y_te"]
    yt = y[2664:]                                   # test split = last 296 samples
    probs = {c: [np.load(f"{S}/probs_{c}_seed{k}.npy") for k in range(5)] for c, _, _ in ABL}
    hist = {}
    for c, _, _ in ABL:
        H = [pickle.load(open(f"{S}/history_{c}_seed{k}.pkl", "rb")) for k in range(5)]
        hist[c] = (np.array([h["loss"] for h in H]), np.array([h["val_loss"] for h in H]))
    return yt, probs, hist


def draw_confusion(ax, yt, probs):
    cm = np.mean([confusion_matrix(yt, (p > 0.5).astype(int), labels=[0, 1]) for p in probs], axis=0)
    rp = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(rp, cmap=CMAP, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.1f}\n{rp[i, j] * 100:.0f}%", ha="center", va="center", fontsize=12,
                    color="white" if rp[i, j] > 0.55 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Severe", "Mild"]); ax.set_yticklabels(["Severe", "Mild"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)


def draw_loss(ax, tr, va, legend_loc):
    ep = np.arange(1, tr.shape[1] + 1)
    for arr, col, ls, lab in [(tr, C_TR, "-", "Training"), (va, C_VA, "--", "Validation")]:
        m, sd = arr.mean(0), arr.std(0, ddof=1)
        ax.fill_between(ep, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.plot(ep, m, color=col, ls=ls, lw=2.0, label=lab)
    ax.set_ylim(0.25, 0.75)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy loss")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc=legend_loc)


def figures678():
    yt, probs, hist = ablation_data()

    def below(ax, text, dy):
        ax.text(0.5, dy, text, transform=ax.transAxes, ha="center", va="top", fontsize=12.5, fontweight="bold")

    # Figure 6: confusion matrices, 2 x 2, label below each panel
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.2))
    for ax, (c, l, name) in zip(axes.ravel(), ABL):
        draw_confusion(ax, yt, probs[c]); below(ax, f"({l}) {name}", -0.30)
    fig.tight_layout(h_pad=3.2); save(fig, "Figure_6.png")

    # Figure 7: ROC curves (mean +/- SD over seeds)
    auc = pd.read_csv(f"{S}/ablation_all_seeds.csv").groupby("config")["test_roc_auc"].mean()
    spec = [("full_fusion", "Full fusion", C_HYB, "-", 2.6), ("esm_metada", "ESM + metadata", C_STACK, "--", 2.0),
            ("no_esm", "No ESM", C_XGB, "-.", 2.0), ("esm_only", "ESM only", C_CNN, ":", 2.0)]
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for c, label, col, ls, lw in spec:
        tprs = np.array([np.interp(grid, *roc_curve(yt, p)[:2]) for p in probs[c]])
        m, sd = tprs.mean(0), tprs.std(0, ddof=1)
        ax.fill_between(grid, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), color=col, alpha=0.15, lw=0)
        ax.plot(grid, m, color=col, ls=ls, lw=lw, label=f"{label}  (AUC {auc[c]:.3f})")
    ax.plot([0, 1], [0, 1], color="#b5b5b0", lw=1.2, ls=(0, (1, 3)), label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout(); save(fig, "Figure_7.png")

    # Figure 8: loss curves, 2 x 2, label below each panel
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.4))
    for ax, (c, l, name) in zip(axes.ravel(), ABL):
        draw_loss(ax, *hist[c], "lower left"); below(ax, f"({l}) {name}", -0.27)
    fig.tight_layout(h_pad=3.2); save(fig, "Figure_8.png")

    # single-panel versions (label in bold under each panel)
    for c, l, name in ABL:
        for kind, size, fn in [("6", (3.8, 3.5), lambda ax: draw_confusion(ax, yt, probs[c])),
                               ("8", (3.8, 3.3), lambda ax: draw_loss(ax, *hist[c], "upper right"))]:
            fig, ax = plt.subplots(figsize=size)
            fn(ax)
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            fig.text(0.5, 0.025, f"({l}) {name}", ha="center", va="center", fontsize=12.5, fontweight="bold")
            path = os.path.join(PAN, f"Figure_{kind}{l}.png")
            fig.savefig(path, dpi=300, facecolor="white"); plt.close(fig)
    print("saved single panels in", PAN)


# ===================================================================
# Tables (mean +/- SD, sample SD over folds or seeds)

def pm(x):
    x = np.asarray(x, float)
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"


def tables(folds):
    # Table 4: experts and stack, five folds
    d = pd.read_csv(f"{CV}/cv_per_fold.csv")
    names = {"cnn": "1D-CNN (Sequence Only)", "xgb": "XGBoost (Metadata)", "hybrid": "Hybrid Model",
             "stack": "Stacked Ensemble (Proposed)"}
    t4 = pd.DataFrame([{"Model": names[m], "Accuracy": pm(g.acc), "F1-Score": pm(g.f1), "ROC-AUC": pm(g.auc)}
                       for m, g in ((m, d[d.model == m]) for m in names)])
    t4.to_csv(f"{TAB}/Table_4.csv", index=False); print("\nTable 4\n", t4.to_string(index=False))

    # Table 5: baselines + hybrid + stack
    bp = find("baselines_per_fold.csv")
    if bp:
        b = pd.read_csv(bp)
        rows = [{"Model": "Stacked ensemble", "Features": "All", "Accuracy": pm(d[d.model == "stack"].acc),
                 "F1-Score": pm(d[d.model == "stack"].f1), "ROC-AUC": pm(d[d.model == "stack"].auc)},
                {"Model": "Hybrid model", "Features": "All", "Accuracy": pm(d[d.model == "hybrid"].acc),
                 "F1-Score": pm(d[d.model == "hybrid"].f1), "ROC-AUC": pm(d[d.model == "hybrid"].auc)}]
        if "XGBoost" not in set(b.model):      # older baseline files have no XGBoost rows; use the stack's expert
            x = d[d.model == "xgb"]
            rows.append({"Model": "XGBoost", "Features": "metadata", "Accuracy": pm(x.acc), "F1-Score": pm(x.f1),
                         "ROC-AUC": pm(x.auc)})
        for (model, feat), g in b.groupby(["model", "features"]):
            rows.append({"Model": model, "Features": feat, "Accuracy": pm(g.acc), "F1-Score": pm(g.f1),
                         "ROC-AUC": pm(g.auc)})
        t5 = pd.DataFrame(rows); t5.to_csv(f"{TAB}/Table_5.csv", index=False)
        print("\nTable 5\n", t5.to_string(index=False))
    else:
        print("\nTable 5 skipped: baselines_per_fold.csv not found")

    # Table 6: seen vs unseen test sequences
    sp = find("seq_ids.npy")
    if sp:
        sid = np.load(sp, allow_pickle=True)
        models = ["stack", "hybrid", "xgb", "cnn"]
        seen, y, P = [], [], {m: [] for m in models}
        for f in folds:
            te = f["te_idx"]
            train_ids = set(sid[np.setdiff1d(np.arange(len(sid)), te)].tolist())
            seen.append(np.array([s in train_ids for s in sid[te]])); y.append(f["y_te"])
            for m in models: P[m].append(f[m])
        seen, y = np.concatenate(seen), np.concatenate(y)
        rows = []
        for m in models:
            p = np.concatenate(P[m]); pred = (p > 0.5).astype(int)
            for nm, mask in [("seen", seen), ("unseen", ~seen)]:
                rows.append({"Model": names[m], "Test samples": nm, "n": int(mask.sum()),
                             "Accuracy": round(accuracy_score(y[mask], pred[mask]), 4),
                             "Macro-F1": round(f1_score(y[mask], pred[mask], average="macro"), 4),
                             "ROC-AUC": round(roc_auc_score(y[mask], p[mask]), 4)})
        t6 = pd.DataFrame(rows); t6.to_csv(f"{TAB}/Table_6.csv", index=False)
        print(f"\nTable 6 (seen {seen.sum()} = {seen.mean():.1%})\n", t6.to_string(index=False))
    else:
        print("\nTable 6 skipped: seq_ids.npy not found")

    # Table 7: sequence-branch ablation, three seeds
    fp = find("ablation_fixed_results.csv")
    if fp:
        r = pd.read_csv(fp)
        lab = {"full": "Full model", "no_transformer": "No Transformer", "fewer_cnn": "Fewer CNN Layers",
               "no_rbd_weight": "No RBD Weight", "no_structure": "No Structure Data",
               "no_hbond": "No H-Bond Data", "no_metadata": "No Metadata"}
        t7 = pd.DataFrame([{"Ablation Configuration": lab[v], "Accuracy": pm(g.accuracy),
                            "F1-Score": pm(g.macro_f1), "ROC-AUC": pm(g.roc_auc)}
                           for v, g in ((v, r[r.variant == v]) for v in lab) if len(g) > 1])
        t7.to_csv(f"{TAB}/Table_7.csv", index=False); print("\nTable 7\n", t7.to_string(index=False))
    else:
        print("\nTable 7 skipped: ablation_fixed_results.csv not found")

    # Table 8: best Optuna trials, sorted by validation F1 (Mild class)
    op = find("optuna_results.csv")
    if op:
        o = pd.read_csv(op).sort_values("val_f1", ascending=False).head(6)
        t8 = pd.DataFrame({"Trial Number": o.trial.astype(int), "Number of CNN Layers": o.num_cnn_layers.astype(int),
                           "Model dimension": o.d_model.astype(int), "CNN Dropout": o.cnn_dropout.round(4),
                           "Transformer Dropout": o.transformer_dropout.round(4),
                           "Accuracy": o.val_accuracy.round(4), "F1-Score": o.val_f1.round(4),
                           "ROC-AUC": pd.to_numeric(o.val_roc_auc, errors="coerce").round(4)})
        t8.to_csv(f"{TAB}/Table_8.csv", index=False); print("\nTable 8\n", t8.to_string(index=False))
    else:
        print("\nTable 8 skipped: optuna_results.csv not found")

    # Table 9: hybrid ablation, five seeds
    a = pd.read_csv(f"{S}/ablation_all_seeds.csv")
    lab = {"full_fusion": "Full Fusion", "no_esm": "No ESM", "esm_metada": "ESM Metadata", "esm_only": "Only ESM"}
    t9 = pd.DataFrame([{"Ablation Configuration": lab[c], "Accuracy": pm(g.test_accuracy),
                        "F1-Score": pm(g.test_f1), "ROC-AUC": pm(g.test_roc_auc)}
                       for c, g in ((c, a[a.config == c]) for c in lab)])
    t9.to_csv(f"{TAB}/Table_9.csv", index=False); print("\nTable 9\n", t9.to_string(index=False))


if __name__ == "__main__":
    figure1()
    figure2()
    folds = figures345()
    figures678()
    tables(folds)
    print("\nDone. Figures in", FIG, "| tables in", TAB)
