#!/usr/bin/env python3
"""
P1 + P2 Combined — Dataset Build + Linear Probes
Traducteur Vectoriel de LLM

1. Build dataset from WikiText-103 subset
2. Tag tokens with spaCy (NER + POS)
3. Extract resid_post activations from GPT-2 small
4. Train linear probes per concept per layer
5. Plot F1-macro by layer
"""

import os
import re
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import spacy
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEVICE = "cpu"  # CUDA crash on this machine — force CPU
MODEL_NAME = "gpt2"  # GPT-2 small, 12 layers, 768 dim
MAX_TOKENS = 5_000  # Total tokens to collect (reduce for speed, increase for quality)
SEQ_LEN = 64  # Tokens per sequence
BATCH_SIZE = 32  # Sequences per batch
N_LAYERS = 12  # GPT-2 small
D_MODEL = 768

# spaCy model
SPACY_MODEL = "en_core_web_sm"

# Concepts to probe
CONCEPTS = {
    "GPE": {"type": "ent", "value": "GPE"},
    "PERSON": {"type": "ent", "value": "PERSON"},
    "ORG": {"type": "ent", "value": "ORG"},
    "DATE": {"type": "ent", "value": "DATE"},
    "NUMBER": {"type": "ent", "value": ["CARDINAL", "ORDINAL", "MONEY", "PERCENT"]},
    "VERB": {"type": "pos", "value": "VERB"},
    "ADJ": {"type": "pos", "value": "ADJ"},
    "NOUN": {"type": "pos", "value": "NOUN"},
    "PROPN": {"type": "pos", "value": "PROPN"},
    "PUNCT": {"type": "pos", "value": "PUNCT"},
}

# Paths
ROOT = Path(__file__).parent.parent.parent  # research/
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

ACTIVATIONS_PATH = DATA_DIR / "activations.pt"
LABELS_PATH = DATA_DIR / "labels.npy"
META_PATH = DATA_DIR / "meta.pkl"
RESULTS_PATH = DATA_DIR / "probe_results.pkl"

# ---------------------------------------------------------------------------
# STEP 1 — LOAD CORPUS
# ---------------------------------------------------------------------------

def load_corpus(max_tokens=MAX_TOKENS, seq_len=SEQ_LEN):
    """Load WikiText-103 raw, filter to clean English paragraphs."""
    print("[P1] Loading WikiText-103 subset...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    paragraphs = []
    total_chars = 0
    target_chars = max_tokens * 4  # rough heuristic: ~4 chars per token

    for item in ds:
        text = item["text"].strip()
        # Skip empty, headings, short junk
        if not text or text.startswith("=") or len(text) < 20:
            continue
        # Skip lines with too much markup
        if "@" in text or "http" in text or "<" in text:
            continue
        paragraphs.append(text)
        total_chars += len(text)
        if total_chars >= target_chars:
            break

    print(f"[P1] Loaded {len(paragraphs)} paragraphs (~{total_chars} chars)")
    return paragraphs

# ---------------------------------------------------------------------------
# STEP 2 — TOKENIZE + TAG
# ---------------------------------------------------------------------------

def build_dataset(paragraphs, model, tokenizer, nlp, max_tokens=MAX_TOKENS):
    """
    For each paragraph:
      - Tokenize with GPT-2 tokenizer (with offsets)
      - Tag with spaCy
      - Align spaCy tokens ↔ GPT-2 tokens via char offsets
      - Extract resid_post activations
    Returns:
      activations: dict layer -> np.array [n_tokens, d_model]
      labels: dict concept -> np.array [n_tokens] (bool)
      token_strings: list of str [n_tokens]
    """
    all_activations = {layer: [] for layer in range(N_LAYERS)}
    all_labels = {concept: [] for concept in CONCEPTS}
    all_token_strings = []
    total_tokens = 0

    # Hook to capture resid_post
    activation_cache = {}

    def hook_fn(layer_name):
        def hook(value, hook, **kwargs):
            # value shape: [batch, pos, d_model]
            # We only run single sequence, so batch=1
            activation_cache[layer_name] = value.detach().cpu()
        return hook

    hooks = [(f"blocks.{l}.hook_resid_post", hook_fn(l)) for l in range(N_LAYERS)]

    print(f"[P1] Processing paragraphs with spaCy + TransformerLens...")
    for para in tqdm(paragraphs, desc="Paragraphs"):
        if total_tokens >= max_tokens:
            break

        # Skip if too long or too short
        if len(para) > 500 or len(para) < 10:
            continue

        # --- GPT-2 tokenization with offsets ---
        encoded = tokenizer(
            para,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=SEQ_LEN,
        )
        gpt_tokens = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        n_gpt = len(gpt_tokens)
        if n_gpt < 5:
            continue

        # --- spaCy tagging ---
        doc = nlp(para)
        spacy_tokens = []
        for tok in doc:
            spacy_tokens.append({
                "start": tok.idx,
                "end": tok.idx + len(tok.text),
                "ent_type": tok.ent_type_,
                "pos": tok.pos_,
                "text": tok.text,
            })

        # --- Align GPT-2 tokens to spaCy labels ---
        token_labels = {c: [] for c in CONCEPTS}
        for i, (start, end) in enumerate(offsets):
            # Find overlapping spaCy token(s)
            matched = False
            for st in spacy_tokens:
                # Overlap condition
                if not (st["end"] <= start or st["start"] >= end):
                    # Match found
                    for concept, cfg in CONCEPTS.items():
                        label = False
                        if cfg["type"] == "ent":
                            if isinstance(cfg["value"], list):
                                label = st["ent_type"] in cfg["value"]
                            else:
                                label = st["ent_type"] == cfg["value"]
                        elif cfg["type"] == "pos":
                            label = st["pos"] == cfg["value"]
                        token_labels[concept].append(label)
                    matched = True
                    break
            if not matched:
                # No spaCy token matched (rare, e.g. whitespace-only GPT token)
                for concept in CONCEPTS:
                    token_labels[concept].append(False)

        # --- Extract activations ---
        input_ids = torch.tensor([gpt_tokens], dtype=torch.long)
        activation_cache.clear()
        with model.hooks(hooks):
            _ = model(input_ids, return_type="logits")

        if not activation_cache:
            continue

        for layer in range(N_LAYERS):
            # shape [1, n_gpt, 768] -> squeeze to [n_gpt, 768]
            acts = activation_cache[layer][0].numpy()  # [n_gpt, 768]
            all_activations[layer].append(acts)

        for concept in CONCEPTS:
            all_labels[concept].extend(token_labels[concept])

        all_token_strings.extend([tokenizer.decode([tid]) for tid in gpt_tokens])
        total_tokens += n_gpt

    # Concatenate
    print(f"[P1] Total tokens collected: {total_tokens}")
    activations = {}
    for layer in range(N_LAYERS):
        if all_activations[layer]:
            activations[layer] = np.concatenate(all_activations[layer], axis=0)
        else:
            activations[layer] = np.zeros((0, D_MODEL), dtype=np.float32)

    labels = {}
    for concept in CONCEPTS:
        labels[concept] = np.array(all_labels[concept], dtype=np.int32)

    return activations, labels, all_token_strings

# ---------------------------------------------------------------------------
# STEP 3 — LINEAR PROBES (P2)
# ---------------------------------------------------------------------------

def train_probes(activations, labels):
    """
    Train a LogisticRegression per concept per layer.
    Returns: results[layer][concept] = {f1, accuracy, support, report}
    """
    results = {layer: {} for layer in range(N_LAYERS)}

    print("[P2] Training linear probes...")
    for layer in tqdm(range(N_LAYERS), desc="Layers"):
        X = activations[layer]  # [N, 768]
        if X.shape[0] == 0:
            continue

        for concept in CONCEPTS:
            y = labels[concept]  # [N]
            if len(y) == 0 or len(np.unique(y)) < 2:
                results[layer][concept] = {"f1": 0.0, "support": 0, "skipped": True}
                continue

            # Check class balance
            n_pos = y.sum()
            n_neg = len(y) - n_pos
            if n_pos < 10 or n_neg < 10:
                results[layer][concept] = {
                    "f1": 0.0,
                    "support": int(n_pos),
                    "skipped": True,
                    "reason": "too few examples",
                }
                continue

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=1,  # CPU-friendly
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
            acc = (y_pred == y_test).mean()

            results[layer][concept] = {
                "f1": float(f1),
                "accuracy": float(acc),
                "support_pos": int(n_pos),
                "support_neg": int(n_neg),
                "coef_norm": float(np.linalg.norm(clf.coef_)),
                "skipped": False,
            }

    return results

# ---------------------------------------------------------------------------
# STEP 4 — PLOT
# ---------------------------------------------------------------------------

def plot_results(results, save_path):
    """Generate F1 heatmap and average F1 by layer."""
    concepts = list(CONCEPTS.keys())
    layers = list(range(N_LAYERS))

    # Build matrix [concept, layer]
    f1_matrix = np.zeros((len(concepts), len(layers)))
    for i, concept in enumerate(concepts):
        for j, layer in enumerate(layers):
            if concept in results[layer] and not results[layer][concept].get("skipped", False):
                f1_matrix[i, j] = results[layer][concept]["f1"]
            else:
                f1_matrix[i, j] = np.nan

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Heatmap
    ax1 = axes[0]
    sns.heatmap(
        f1_matrix,
        xticklabels=[f"L{l}" for l in layers],
        yticklabels=concepts,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1.0,
        ax=ax1,
        cbar_kws={"label": "F1 (binary)"},
    )
    ax1.set_title("Linear Probe F1 by Concept and Layer — GPT-2 small", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Concept")

    # Average F1 per layer
    ax2 = axes[1]
    avg_f1 = np.nanmean(f1_matrix, axis=0)
    ax2.plot(layers, avg_f1, marker="o", linewidth=2, markersize=8, color="#2E86AB")
    ax2.axhline(0.65, color="red", linestyle="--", alpha=0.6, label="F1 = 0.65 threshold")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean F1 (macro)")
    ax2.set_title("Mean F1 Across Concepts by Layer")
    ax2.set_xticks(layers)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[P2] Figure saved: {save_path}")
    plt.close()

    # Best layer
    best_layer = int(np.nanargmax(avg_f1))
    print(f"[P2] Best layer (highest mean F1): {best_layer} (F1={avg_f1[best_layer]:.3f})")
    print(f"[P2] Concepts with F1 > 0.65 at best layer:")
    for concept in concepts:
        f1_val = f1_matrix[concepts.index(concept), best_layer]
        if f1_val > 0.65:
            print(f"    [OK] {concept}: F1={f1_val:.3f}")

    return best_layer, avg_f1

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("TRADUCTEUR VECTORIEL — P1 Dataset + P2 Linear Probes")
    print("=" * 60)

    # Load spaCy
    print(f"[Setup] Loading spaCy model: {SPACY_MODEL}...")
    nlp = spacy.load(SPACY_MODEL)

    # Load GPT-2 small via TransformerLens
    print(f"[Setup] Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tokenizer = model.tokenizer
    model.eval()
    print(f"[Setup] Model loaded: {MODEL_NAME} | Layers: {N_LAYERS} | D_model: {D_MODEL}")

    # P1 — Build dataset
    if ACTIVATIONS_PATH.exists() and LABELS_PATH.exists():
        print(f"[P1] Cached dataset found. Loading...")
        activations = torch.load(ACTIVATIONS_PATH, weights_only=False)
        # Convert to numpy if needed
        activations = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in activations.items()}
        labels = np.load(LABELS_PATH, allow_pickle=True).item()
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        token_strings = meta["token_strings"]
        print(f"[P1] Loaded cached: {len(token_strings)} tokens")
    else:
        paragraphs = load_corpus(max_tokens=MAX_TOKENS)
        activations, labels, token_strings = build_dataset(paragraphs, model, tokenizer, nlp, max_tokens=MAX_TOKENS)

        # Save
        torch.save(activations, ACTIVATIONS_PATH)
        np.save(LABELS_PATH, labels)
        with open(META_PATH, "wb") as f:
            pickle.dump({"token_strings": token_strings, "n_layers": N_LAYERS, "d_model": D_MODEL}, f)
        print(f"[P1] Dataset saved to {DATA_DIR}")

    # P2 — Train probes
    if RESULTS_PATH.exists():
        print(f"[P2] Cached results found. Loading...")
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
    else:
        results = train_probes(activations, labels)
        with open(RESULTS_PATH, "wb") as f:
            pickle.dump(results, f)
        print(f"[P2] Results saved to {RESULTS_PATH}")

    # Plot
    fig_path = FIG_DIR / "p2_f1_by_layer.png"
    best_layer, avg_f1 = plot_results(results, fig_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tokens in dataset: {len(token_strings)}")
    print(f"Best layer for decoding: {best_layer} (mean F1 = {avg_f1[best_layer]:.3f})")
    print(f"Figure: {fig_path}")
    print(f"Data cache: {DATA_DIR}")
    print("=" * 60)

    # Update tracker info
    print("\n[Next] P3: Use layer {} as input to the MLP translator.".format(best_layer))

if __name__ == "__main__":
    main()
