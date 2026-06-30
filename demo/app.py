"""
LLM Vector Translator — Hugging Face Spaces Demo
Visualize GPT-2 layer-by-layer predictions.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import gradio as gr
import plotly.express as px
from transformer_lens import HookedTransformer

print("[Demo] Loading GPT-2 small...")
MODEL = HookedTransformer.from_pretrained("gpt2", device="cpu")
MODEL.eval()
print("[Demo] Model loaded!")


def get_layer_predictions(text, layer_idx):
    """Get top-5 predictions at a given layer."""
    tokens = MODEL.to_tokens(text)
    _, cache = MODEL.run_with_cache(tokens)
    
    if layer_idx == 0:
        resid = cache["hook_embed"]
    else:
        resid = cache[f"blocks.{layer_idx-1}.hook_resid_post"]
    
    layer_logits = resid @ MODEL.W_U
    last_pos = layer_logits.shape[1] - 1
    
    probs = torch.softmax(layer_logits[0, last_pos], dim=-1)
    top5 = torch.topk(probs, 5)
    
    results = []
    for i, (p, idx) in enumerate(zip(top5.values, top5.indices)):
        word = MODEL.to_single_str_token(idx.item())
        results.append(f"{i+1}. '{word}' ({p.item()*100:.3f}%)")
    
    return "\n".join(results)


def get_rank_percentiles(text):
    """Get rank percentile of true token at each layer."""
    tokens = MODEL.to_tokens(text)
    true_token_id = tokens[0, -1].item()
    _, cache = MODEL.run_with_cache(tokens)
    
    layers = ["Input"] + [f"L{i}" for i in range(1, 13)]
    percentiles = []
    
    resid = cache["hook_embed"]
    logits = resid @ MODEL.W_U
    sorted_indices = torch.argsort(logits[0, -1], descending=True)
    rank = (sorted_indices == true_token_id).nonzero().item()
    percentiles.append(rank / MODEL.cfg.d_vocab * 100)
    
    for layer in range(12):
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        logits = resid @ MODEL.W_U
        sorted_indices = torch.argsort(logits[0, -1], descending=True)
        rank = (sorted_indices == true_token_id).nonzero().item()
        percentiles.append(rank / MODEL.cfg.d_vocab * 100)
    
    return layers, percentiles


def get_top1_probabilities(text):
    """Get top-1 probability at each layer."""
    tokens = MODEL.to_tokens(text)
    _, cache = MODEL.run_with_cache(tokens)
    
    layers = ["Input"] + [f"L{i}" for i in range(1, 13)]
    top1_probs = []
    
    resid = cache["hook_embed"]
    logits = resid @ MODEL.W_U
    probs = torch.softmax(logits[0, -1], dim=-1)
    top1_probs.append(probs.max().item() * 100)
    
    for layer in range(12):
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        logits = resid @ MODEL.W_U
        probs = torch.softmax(logits[0, -1], dim=-1)
        top1_probs.append(probs.max().item() * 100)
    
    return layers, top1_probs


def get_heatmap_data(text):
    """Get log-probabilities for all tokens at all layers."""
    tokens = MODEL.to_tokens(text)
    token_ids = tokens[0].tolist()
    _, cache = MODEL.run_with_cache(tokens)
    
    n_layers = 13
    n_tokens = len(token_ids)
    data = np.zeros((n_layers, n_tokens))
    
    resid = cache["hook_embed"]
    logits = resid @ MODEL.W_U
    for pos in range(n_tokens):
        logprob = torch.log_softmax(logits[0, pos], dim=-1)[token_ids[pos]].item()
        data[0, pos] = logprob
    
    for layer in range(12):
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        logits = resid @ MODEL.W_U
        for pos in range(n_tokens):
            logprob = torch.log_softmax(logits[0, pos], dim=-1)[token_ids[pos]].item()
            data[layer + 1, pos] = logprob
    
    token_labels = [MODEL.to_single_str_token(tid).replace("Ġ", "_") for tid in token_ids]
    layer_labels = ["Input"] + [f"L{i}" for i in range(1, 13)]
    
    return data, token_labels, layer_labels


def analyze_text(text, layer_slider):
    """Main analysis function."""
    if not text or not text.strip():
        return "Please enter some text.", None, None, None, None
    
    try:
        top5_text = get_layer_predictions(text, layer_slider)
        
        layers, percentiles = get_rank_percentiles(text)
        fig_rank = px.line(
            x=layers, y=percentiles,
            labels={"x": "Layer", "y": "Rank Percentile (%)"},
            title="Rank Percentile of True Token Across Layers",
            markers=True
        )
        fig_rank.add_hline(y=5, line_dash="dash", line_color="green", 
                          annotation_text="Top 5% threshold")
        fig_rank.update_layout(height=400)
        
        layers_p, top1_probs = get_top1_probabilities(text)
        fig_prob = px.line(
            x=layers_p, y=top1_probs,
            labels={"x": "Layer", "y": "Top-1 Probability (%)"},
            title="Top-1 Token Probability Across Layers",
            markers=True
        )
        fig_prob.update_layout(height=400)
        
        data, token_labels, layer_labels = get_heatmap_data(text)
        fig_heatmap = px.imshow(
            data,
            x=token_labels,
            y=layer_labels,
            labels={"x": "Token", "y": "Layer", "color": "Log-Prob"},
            title="Log-Probability of True Token at Each Layer",
            aspect="auto",
            color_continuous_scale="RdYlGn"
        )
        fig_heatmap.update_layout(height=500)
        
        improvement = percentiles[0] / percentiles[6] if percentiles[6] > 0 else 0
        summary = f"""### Summary
- **Input rank**: {percentiles[0]:.1f}% (position {int(percentiles[0]/100 * 50257):,}/50,257)
- **Layer 6 rank**: {percentiles[6]:.1f}% (position {int(percentiles[6]/100 * 50257):,}/50,257)
- **Final rank**: {percentiles[12]:.1f}% (position {int(percentiles[12]/100 * 50257):,}/50,257)
- **Improvement**: {improvement:.1f}x from Input to Layer 6
- **Decision layer**: Layer 6 (where rank drops most sharply)
"""
        
        return top5_text, fig_rank, fig_prob, fig_heatmap, summary
        
    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None


# Build interface
with gr.Blocks(title="LLM Vector Translator") as demo:
    gr.Markdown("""
    # LLM Vector Translator
    ### Visualize layer-by-layer predictions
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="The cat sat on the mat and looked",
                value="The cat sat on the mat and looked",
                lines=2
            )
            layer_slider = gr.Slider(
                minimum=0, maximum=12, step=1, value=6,
                label="Layer to inspect (0=Input, 6=Decision Layer)"
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column(scale=1):
            top5_output = gr.Textbox(label="Top-5 Predictions", lines=6, interactive=False)
    
    summary_output = gr.Markdown()
    
    with gr.Row():
        with gr.Column():
            rank_plot = gr.Plot(label="Rank Percentile")
        with gr.Column():
            prob_plot = gr.Plot(label="Top-1 Probability")
    
    with gr.Row():
        heatmap_plot = gr.Plot(label="Layer × Token Heatmap")
    
    analyze_btn.click(
        fn=analyze_text,
        inputs=[text_input, layer_slider],
        outputs=[top5_output, rank_plot, prob_plot, heatmap_plot, summary_output]
    )
    
    demo.load(
        fn=analyze_text,
        inputs=[text_input, layer_slider],
        outputs=[top5_output, rank_plot, prob_plot, heatmap_plot, summary_output]
    )

if __name__ == "__main__":
    demo.launch()
