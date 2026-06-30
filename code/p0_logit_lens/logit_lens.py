"""
P0 — Logit Lens sur GPT-2 small
================================
Objectif : extraire les activations du residual stream à chaque couche
et visualiser comment la prédiction du prochain token se forme progressivement.

Usage :
    python logit_lens.py

Dépendances :
    pip install transformer_lens matplotlib seaborn numpy torch

Sortie :
    - Console : tableau token prédit + probabilité par couche
    - Fichier  : logit_lens_heatmap.png (sauvegardé dans ../../figures/)
"""

import re
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # mode non-interactif (changer en "TkAgg" si tu veux une fenêtre)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import transformer_lens

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_NAME   = "gpt2"          # GPT-2 small (124M), ~0.3 Go fp16
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
FIGURES_DIR  = Path(__file__).parent.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Phrases de test — varier les types pour observer des comportements différents
TEST_PROMPTS = [
    "The Eiffel Tower is located in",
    "def fibonacci(n):\n    if n <= 1:\n        return",
    "The capital of Japan is",
    "Once upon a time there was a",
    "2 + 2 =",
]

TOP_K = 5  # nombre de tokens à afficher dans le console pour chaque couche


# ─────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────
def load_model(model_name: str, device: str):
    """Charge le modèle via TransformerLens."""
    print(f"[+] Chargement de {model_name} sur {device}...")
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,     # centre la matrice d'embedding pour stabiliser logit lens
        center_writing_weights=True,
        fold_ln=True,            # fusionne les LayerNorm dans les poids — accélère logit lens
        device=device,
    )
    model.eval()
    print(f"    [OK] {model.cfg.n_layers} couches, d_model={model.cfg.d_model}")
    return model


# ─────────────────────────────────────────────
# LOGIT LENS
# ─────────────────────────────────────────────
def logit_lens(model, prompt: str):
    """
    Pour chaque couche l, projette resid_post_l via l'unembed
    et retourne le token le plus probable + sa probabilité.

    Retourne :
        tokens_input  : list[str]  — tokens du prompt
        top_tokens    : list[list[str]]  — shape [n_layers+1, seq_len]
        top_probs     : ndarray  — shape [n_layers+1, seq_len]
    """
    tokens = model.to_tokens(prompt)  # shape [1, seq_len]
    n_layers = model.cfg.n_layers

    # forward pass avec capture de toutes les activations
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    seq_len = tokens.shape[1]
    token_strings = [model.to_string(tokens[0, i]) for i in range(seq_len)]

    top_tokens = []
    top_probs  = []

    # couches 0 → n_layers - 1
    for layer in range(n_layers):
        resid = cache["resid_post", layer]  # [1, seq_len, d_model]
        # application manuelle du layer_norm final + unembed
        resid_ln = model.ln_final(resid)
        logits = model.unembed(resid_ln)    # [1, seq_len, vocab_size]
        probs  = torch.softmax(logits, dim=-1)

        top1_prob, top1_idx = probs[0].max(dim=-1)  # [seq_len]
        layer_tokens = [model.to_string(top1_idx[i]) for i in range(seq_len)]
        top_tokens.append(layer_tokens)
        top_probs.append(top1_prob.detach().cpu().numpy())

    # couche finale (sortie réelle du modèle)
    final_logits = model(tokens)  # [1, seq_len, vocab_size]
    final_probs  = torch.softmax(final_logits, dim=-1)
    top1_prob, top1_idx = final_probs[0].max(dim=-1)
    top_tokens.append([model.to_string(top1_idx[i]) for i in range(seq_len)])
    top_probs.append(top1_prob.detach().cpu().numpy())

    return token_strings, top_tokens, np.array(top_probs)


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
def plot_logit_lens(token_strings, top_tokens, top_probs, prompt: str, save_path: Path):
    """
    Heatmap : X = position dans la séquence, Y = couche, couleur = proba du top-1 token.
    Le texte dans chaque cellule = le token prédit.
    """
    n_layers_plus1, seq_len = top_probs.shape
    layer_labels = [f"L{i}" for i in range(n_layers_plus1 - 1)] + ["Final"]

    fig, ax = plt.subplots(figsize=(max(8, seq_len * 1.2), max(6, n_layers_plus1 * 0.55)))

    # heatmap des probabilités
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        top_probs,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor="gray",
        cbar_kws={"label": "P(top-1 token)", "shrink": 0.6},
        annot=False,
    )

    # texte des tokens prédits dans chaque cellule
    for layer in range(n_layers_plus1):
        for pos in range(seq_len):
            token_txt = top_tokens[layer][pos].replace("\n", "\\n").strip()
            prob = top_probs[layer, pos]
            color = "black" if prob < 0.7 else "white"
            ax.text(
                pos + 0.5, layer + 0.5,
                token_txt[:8],  # tronquer pour éviter le débordement
                ha="center", va="center",
                fontsize=7, color=color, fontweight="bold"
            )

    # axes
    ax.set_xticks(np.arange(seq_len) + 0.5)
    ax.set_xticklabels(
        [f'"{t.strip()}"' for t in token_strings],
        rotation=45, ha="right", fontsize=9
    )
    ax.set_yticks(np.arange(n_layers_plus1) + 0.5)
    ax.set_yticklabels(layer_labels, fontsize=8)
    ax.set_xlabel("Token d'entrée (position)", fontsize=10)
    ax.set_ylabel("Couche", fontsize=10)
    ax.set_title(f'Logit Lens — "{prompt[:60]}{"..." if len(prompt) > 60 else ""}"',
                 fontsize=11, pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [OK] Figure sauvegardée : {save_path}")


def print_layer_summary(token_strings, top_tokens, top_probs, prompt: str):
    """Affiche un résumé console couche par couche pour le dernier token."""
    n_layers = len(top_tokens) - 1
    last_pos  = len(token_strings) - 1

    print(f"\n{'-'*60}")
    print(f"Prompt : <<{prompt}>>")
    print(f"Token prédit pour la position finale <<{token_strings[last_pos]}>>")
    print(f"{'-'*60}")
    print(f"{'Couche':<10} {'Token prédit':<20} {'Proba':<8}")
    print(f"{'-'*10} {'-'*20} {'-'*8}")

    for layer in range(n_layers + 1):
        label = f"L{layer}" if layer < n_layers else "Final"
        tok   = top_tokens[layer][last_pos]
        prob  = top_probs[layer, last_pos]
        marker = " <- OK" if layer == n_layers else ""
        print(f"{label:<10} {repr(tok):<20} {prob:.4f}{marker}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    model = load_model(MODEL_NAME, DEVICE)

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{i+1}/{len(TEST_PROMPTS)}] Traitement : «{prompt}»")

        token_strings, top_tokens, top_probs = logit_lens(model, prompt)

        # résumé console
        print_layer_summary(token_strings, top_tokens, top_probs, prompt)

        # figure
        safe_name = re.sub(r'[^\w\-_.]', '_', prompt[:40]) + ".png"
        save_path = FIGURES_DIR / f"logit_lens_{i+1}_{safe_name}"
        plot_logit_lens(token_strings, top_tokens, top_probs, prompt, save_path)

    print("\n[DONE] Phase P0 terminée. Vérifier les figures dans research/figures/")
    print("    Critère de succès : la bonne prédiction apparaît dans les couches 9–11.")


if __name__ == "__main__":
    main()
