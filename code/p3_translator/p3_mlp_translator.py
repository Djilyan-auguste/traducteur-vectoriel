#!/usr/bin/env python3
"""
P3 — Traducteur MLP (Multi-Label Concept Translator)
Traducteur Vectoriel de LLM

Test architectures A (768->128->10) et B (768->512->256->10)
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
BEST_LAYER = 6
N_CONCEPTS = 10
D_MODEL = 768
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 5

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

ACTIVATIONS_PATH = DATA_DIR / "activations.pt"
LABELS_PATH = DATA_DIR / "labels.npy"
META_PATH = DATA_DIR / "meta.pkl"
P2_RESULTS_PATH = DATA_DIR / "probe_results.pkl"

CONCEPTS = ["GPE", "PERSON", "ORG", "DATE", "NUMBER", "VERB", "ADJ", "NOUN", "PROPN", "PUNCT"]

# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------

class ModelA(nn.Module):
    """Option A: 768 -> 128 -> 10 (petit, moins de params)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(128, N_CONCEPTS)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ModelB(nn.Module):
    """Option B: 768 -> 512 -> 256 -> 10 (profond, plus expressif)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(DROPOUT)
        self.fc3 = nn.Linear(256, N_CONCEPTS)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data():
    print("[P3] Loading P2 data...")
    activations_dict = torch.load(ACTIVATIONS_PATH, weights_only=False)
    activations_dict = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in activations_dict.items()}
    X = activations_dict[BEST_LAYER]
    labels_dict = np.load(LABELS_PATH, allow_pickle=True).item()
    y = np.stack([labels_dict[c] for c in CONCEPTS], axis=1).astype(np.float32)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    print(f"[P3] Loaded: X={X.shape}, y={y.shape}")
    return X, y

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": [], "val_f1_macro": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_true = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float().cpu().numpy()
                all_val_preds.append(preds)
                all_val_true.append(yb.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_preds = np.concatenate(all_val_preds, axis=0)
        val_true = np.concatenate(all_val_true, axis=0)
        val_f1_macro = f1_score(val_true, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1_macro"].append(val_f1_macro)

        print(f"Epoch {epoch:03d} | train={avg_train_loss:.4f} | val={avg_val_loss:.4f} | val_f1={val_f1_macro:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"[P3] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader):
    model.eval()
    all_probs = []
    all_preds = []
    all_true = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            logits = model(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(np.float32)
            all_probs.append(probs)
            all_preds.append(preds)
            all_true.append(yb.numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    true = np.concatenate(all_true, axis=0)

    per_concept_f1 = {}
    for i, concept in enumerate(CONCEPTS):
        per_concept_f1[concept] = float(f1_score(true[:, i], preds[:, i], zero_division=0))

    macro_f1 = f1_score(true, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(true, preds, average="micro", zero_division=0)
    avg_precision = average_precision_score(true, probs, average="macro")

    print(f"\nTest Metrics: Macro F1={macro_f1:.4f} | Micro F1={micro_f1:.4f} | AvgPrec={avg_precision:.4f}")
    for c in CONCEPTS:
        print(f"  {c:12s}: {per_concept_f1[c]:.4f}")

    return {
        "per_concept_f1": per_concept_f1,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "avg_precision": float(avg_precision),
        "probs": probs, "preds": preds, "true": true,
    }

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_results(history, metrics, p2_results, save_prefix):
    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax.plot(epochs, history["val_f1_macro"], label="Val F1 Macro", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / F1")
    ax.set_title(f"Training History — {save_prefix.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_history.png", dpi=150, bbox_inches="tight")
    plt.close()

    # MLP vs Linear
    p2_f1 = {}
    if p2_results and BEST_LAYER in p2_results:
        for c in CONCEPTS:
            p2_f1[c] = p2_results[BEST_LAYER][c]["f1"] if c in p2_results[BEST_LAYER] and not p2_results[BEST_LAYER][c].get("skipped") else 0.0

    x = np.arange(len(CONCEPTS))
    fig, ax = plt.subplots(figsize=(12, 6))
    p2_vals = [p2_f1.get(c, 0.0) for c in CONCEPTS]
    p3_vals = [metrics["per_concept_f1"][c] for c in CONCEPTS]
    ax.bar(x - 0.2, p2_vals, 0.4, label="Linear Probe (P2)", color="#2E86AB", alpha=0.8)
    ax.bar(x + 0.2, p3_vals, 0.4, label="MLP (P3)", color="#A23B72", alpha=0.8)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Concept Decoding — {save_prefix.name}")
    ax.set_xticks(x)
    ax.set_xticklabels(CONCEPTS, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.65, color="red", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_vs_linear.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_experiment(model_class, name, suffix, train_loader, val_loader, test_loader, p2_results):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {name}")
    print("=" * 60)
    model = model_class().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    model, history = train_model(model, train_loader, val_loader)
    metrics = evaluate_model(model, test_loader)

    torch.save(model.state_dict(), DATA_DIR / f"p3_model_{suffix}.pt")
    plot_results(history, metrics, p2_results, FIG_DIR / f"p3_{suffix}")

    print(f"\nSUMMARY {name}: Macro F1={metrics['macro_f1']:.4f} | Micro F1={metrics['micro_f1']:.4f}")
    return metrics, n_params

def main():
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("P3 — Test A (128) et B (512->256)")
    print("=" * 60)

    X, y = load_data()

    # Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.111, random_state=42)
    print(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Datasets
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load P2 results
    p2_results = None
    if P2_RESULTS_PATH.exists():
        with open(P2_RESULTS_PATH, "rb") as f:
            p2_results = pickle.load(f)

    # Run A and B
    results_a, params_a = run_experiment(ModelA, "A: 768->128->10", "A_128", train_loader, val_loader, test_loader, p2_results)
    results_b, params_b = run_experiment(ModelB, "B: 768->512->256->10", "B_512_256", train_loader, val_loader, test_loader, p2_results)

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} | {'Params':<10} | {'Macro F1':<10} | {'Micro F1':<10}")
    print("-" * 60)
    print(f"{'A (128)':<25} | {params_a:<10,} | {results_a['macro_f1']:<10.4f} | {results_a['micro_f1']:<10.4f}")
    print(f"{'B (512->256)':<25} | {params_b:<10,} | {results_b['macro_f1']:<10.4f} | {results_b['micro_f1']:<10.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
