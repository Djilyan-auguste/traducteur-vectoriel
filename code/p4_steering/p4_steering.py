#!/usr/bin/env python3
"""
P4 — Validation Causale par Activation Steering
Traducteur Vectoriel de LLM

1. Extrait les "directions de concept" depuis les activations P1
   (mean-diff : moyenne(pos) - moyenne(neg))
2. Injecte ces directions dans le residual stream de GPT-2 small
   à la couche 6 (meilleure couche P2/P3)
3. Mesure le virage comportemental : proba des tokens du concept
   avant vs après steering
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEVICE = "cpu"
MODEL_NAME = "gpt2"
BEST_LAYER = 6
N_LAYENS = 12
D_MODEL = 768

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

ACTIVATIONS_PATH = DATA_DIR / "activations.pt"
LABELS_PATH = DATA_DIR / "labels.npy"
META_PATH = DATA_DIR / "meta.pkl"

# Concepts à tester (ceux avec F1 > 0.65 dans P3)
CONCEPTS = ["DATE", "NUMBER", "NOUN", "PROPN", "VERB", "ORG"]

# Prompts de test neutres — on va voir si le steering les "incline"
# vers le concept cible
TEST_PROMPTS = {
    "DATE": [
        "The meeting is scheduled for",
        "She was born in",
        "The treaty was signed on",
    ],
    "NUMBER": [
        "The population of the city is",
        "He scored",
        "The answer is",
    ],
    "NOUN": [
        "The",
        "A beautiful",
        "In the",
    ],
    "PROPN": [
        "The capital of France is",
        "Founded by",
        "According to",
    ],
    "VERB": [
        "She",
        "The team",
        "They",
    ],
    "ORG": [
        "The company",
        "Founded in 2010,",
        "The organization",
    ],
}

# Tokens cibles associés à chaque concept (pour mesurer la proba)
# Ce sont des tokens que le modèle devrait favoriser si le steering marche
TARGET_TOKENS = {
    "DATE": [" January", " February", " March", " 2024", " 2025", " Monday", " Tuesday"],
    "NUMBER": [" 100", " 1000", " 42", " 7", " 50", " 10", " 1"],
    "NOUN": [" dog", " cat", " house", " car", " book", " tree", " water"],
    "PROPN": [" Paris", " London", " John", " Alice", " France", " Google", " Tesla"],
    "VERB": [" runs", " walks", " eats", " sleeps", " reads", " writes", " jumps"],
    "ORG": [" Google", " Microsoft", " Apple", " Amazon", " Tesla", " Facebook", " Netflix"],
}

# Coefficients de steering à tester
ALPHAS = np.linspace(-20, 20, 9)  # -20, -15, -10, -5, 0, 5, 10, 15, 20

# ---------------------------------------------------------------------------
# 1 — EXTRACTION DES DIRECTIONS
# ---------------------------------------------------------------------------

def load_data():
    """Charge activations, labels, meta."""
    print("[P4] Loading P1/P2 data...")
    activations_dict = torch.load(ACTIVATIONS_PATH, weights_only=False)
    activations_dict = {
        k: v.numpy() if isinstance(v, torch.Tensor) else v
        for k, v in activations_dict.items()
    }
    X = activations_dict[BEST_LAYER]  # [N, 768]
    labels_dict = np.load(LABELS_PATH, allow_pickle=True).item()
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    token_strings = meta["token_strings"]
    print(f"[P4] Loaded: X={X.shape}, tokens={len(token_strings)}")
    return X, labels_dict, token_strings


def extract_directions(X, labels_dict):
    """
    Pour chaque concept, calcule la direction = mean(pos) - mean(neg).
    Retourne dict concept -> np.array(768,)
    """
    directions = {}
    for concept in CONCEPTS:
        y = labels_dict[concept]  # [N] int32 0/1
        pos_mask = y == 1
        neg_mask = y == 0
        if pos_mask.sum() < 5 or neg_mask.sum() < 5:
            print(f"  [SKIP] {concept}: too few examples (pos={pos_mask.sum()}, neg={neg_mask.sum()})")
            continue
        mean_pos = X[pos_mask].mean(axis=0)  # [768]
        mean_neg = X[neg_mask].mean(axis=0)  # [768]
        direction = mean_pos - mean_neg
        # Normalize
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        directions[concept] = direction
        print(f"  [OK] {concept}: pos={pos_mask.sum()}, neg={neg_mask.sum()}, dir_norm={np.linalg.norm(direction):.4f}")
    return directions


# ---------------------------------------------------------------------------
# 2 — STEERING HOOK
# ---------------------------------------------------------------------------

def make_steering_hook(direction, alpha, layer=BEST_LAYER):
    """
    Crée un hook TransformerLens qui injecte alpha * direction
    dans le residual stream à la couche donnée.
    """
    direction_t = torch.tensor(direction, dtype=torch.float32).to(DEVICE)

    def hook_fn(value, hook):
        # value shape: [batch, pos, d_model]
        # On ajoute alpha * direction à TOUS les tokens de la séquence
        value += alpha * direction_t
        return value

    return (f"blocks.{layer}.hook_resid_post", hook_fn)


# ---------------------------------------------------------------------------
# 3 — MESURE DU VIRAGE COMPORTEMENTAL
# ---------------------------------------------------------------------------

def measure_steering_effect(model, tokenizer, prompt, concept, direction, alphas):
    """
    Pour un prompt donné, mesure la proba cumulée des tokens cibles
    du concept, pour chaque alpha.
    Retourne : dict alpha -> proba_cible
    """
    # Tokenize
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    tokens_list = tokens[0].tolist()
    last_pos = len(tokens_list) - 1

    target_strings = TARGET_TOKENS[concept]
    # Convertir en token IDs
    target_ids = []
    for ts in target_strings:
        ids = tokenizer.encode(ts, add_special_tokens=False)
        if ids:
            target_ids.extend(ids)
    target_ids = list(set(target_ids))  # dédupliquer

    if not target_ids:
        return {}

    results = {}
    for alpha in alphas:
        hook = make_steering_hook(direction, alpha, layer=BEST_LAYER)
        with model.hooks([hook]):
            logits = model(tokens, return_type="logits")  # [1, seq_len, vocab]

        # Logits pour le prochain token (position last_pos)
        next_logits = logits[0, last_pos, :]  # [vocab]
        probs = torch.softmax(next_logits, dim=-1).detach().cpu().numpy()

        # Proba cumulée des tokens cibles
        target_prob = sum(probs[tid] for tid in target_ids)
        results[float(alpha)] = float(target_prob)

    return results


def run_steering_experiment(model, tokenizer, directions):
    """
    Pour chaque concept et chaque prompt de test, mesure l'effet du steering.
    Retourne : results[concept][prompt] = dict(alpha -> prob)
    """
    results = {c: {} for c in CONCEPTS}

    for concept in CONCEPTS:
        if concept not in directions:
            continue
        direction = directions[concept]
        prompts = TEST_PROMPTS.get(concept, [])
        print(f"\n[P4] Steering concept: {concept}")

        for prompt in tqdm(prompts, desc=f"  {concept} prompts"):
            prob_by_alpha = measure_steering_effect(
                model, tokenizer, prompt, concept, direction, ALPHAS
            )
            results[concept][prompt] = prob_by_alpha

    return results


# ---------------------------------------------------------------------------
# 4 — VISUALISATION
# ---------------------------------------------------------------------------

def plot_steering_results(results, save_dir):
    """
    Génère une figure par concept : proba cible en fonction d'alpha,
    avec une courbe par prompt.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, concept in enumerate(CONCEPTS):
        ax = axes[idx]
        if concept not in results or not results[concept]:
            ax.set_title(f"{concept} — NO DATA")
            continue

        for prompt, prob_by_alpha in results[concept].items():
            if not prob_by_alpha:
                continue
            alphas_sorted = sorted(prob_by_alpha.keys())
            probs_sorted = [prob_by_alpha[a] for a in alphas_sorted]
            # Tronquer le prompt pour la légende
            label = prompt[:30] + "..." if len(prompt) > 30 else prompt
            ax.plot(alphas_sorted, probs_sorted, marker="o", label=label, linewidth=2)

        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Steering coefficient alpha")
        ax.set_ylabel("P(target token)")
        ax.set_title(f"Concept: {concept}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle("Activation Steering — Effect on Target Token Probability", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = save_dir / "p4_steering_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[P4] Figure saved: {save_path}")
    plt.close()


def print_summary_table(results):
    """Affiche un tableau avant/après pour alpha=±10."""
    print("\n" + "=" * 80)
    print("STEERING SUMMARY — Target Token Probability")
    print("=" * 80)
    print(f"{'Concept':<12} | {'Prompt':<35} | {'alpha=0':<10} | {'alpha=-10':<10} | {'alpha=+10':<10} | {'Delta(+-10)':<10}")
    print("-" * 80)

    for concept in CONCEPTS:
        if concept not in results:
            continue
        for prompt, prob_by_alpha in results[concept].items():
            if not prob_by_alpha:
                continue
            p0 = prob_by_alpha.get(0.0, 0.0)
            p_neg = prob_by_alpha.get(-10.0, 0.0)
            p_pos = prob_by_alpha.get(10.0, 0.0)
            delta = p_pos - p_neg
            prompt_short = (prompt[:32] + "...") if len(prompt) > 32 else prompt
            print(f"{concept:<12} | {prompt_short:<35} | {p0:<10.4f} | {p_neg:<10.4f} | {p_pos:<10.4f} | {delta:<+10.4f}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("P4 — Activation Steering (Causal Validation)")
    print("=" * 60)

    # 1. Load data
    X, labels_dict, token_strings = load_data()

    # 2. Extract concept directions
    print("\n[P4] Extracting concept directions (mean-diff)...")
    directions = extract_directions(X, labels_dict)
    print(f"[P4] Directions extracted: {list(directions.keys())}")

    # 3. Load GPT-2
    print(f"\n[P4] Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tokenizer = model.tokenizer
    model.eval()
    print(f"[P4] Model loaded on {DEVICE}")

    # 4. Run steering experiment
    print(f"\n[P4] Running steering with alpha in {ALPHAS.tolist()}")
    results = run_steering_experiment(model, tokenizer, directions)

    # 5. Save results
    results_path = DATA_DIR / "p4_steering_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n[P4] Results saved: {results_path}")

    # 6. Plot + summary
    plot_steering_results(results, FIG_DIR)
    print_summary_table(results)

    # 7. Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
Si le steering est causal :
  -> alpha > 0 augmente P(target token)  (la direction pousse vers le concept)
  -> alpha < 0 diminue P(target token)   (la direction repousse du concept)
  -> La courbe est monotone croissante avec alpha

Si le steering n'est pas causal :
  -> alpha n'a pas d'effet systématique
  -> Courbe plate ou bruitée
  -> Le concept n'est pas linérairement encodé dans cette direction
    """)

    print("=" * 60)
    print("P4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
