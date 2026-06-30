from transformers import GPT2Tokenizer
import spacy

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
nlp = spacy.load("en_core_web_sm")

text = 'Marie Curie was born in Paris on November 7, 1867. She was a physicist and chemist.'

# GPT-2 tokens avec offsets
encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
gpt_tokens = encoded['input_ids']
gpt_offsets = encoded['offset_mapping']

# spaCy tokens avec offsets
doc = nlp(text)
spacy_tokens = []
for tok in doc:
    spacy_tokens.append({
        'text': tok.text,
        'start': tok.idx,
        'end': tok.idx + len(tok.text),
        'pos': tok.pos_,
        'ent': tok.ent_type_ if tok.ent_type_ else None
    })

print('=== ALIGNEMENT GPT-2 - SPACY ===')
print(f"{'GPT Token':15s} | {'Offset':10s} | {'POS':8s} | {'Entity':10s}")
print("-" * 55)

for i, (token_id, (g_start, g_end)) in enumerate(zip(gpt_tokens, gpt_offsets)):
    gpt_str = tokenizer.decode([token_id])
    
    # Trouver le spaCy token qui chevauche cet offset GPT-2
    matched_pos = "—"
    matched_ent = "—"
    
    for st in spacy_tokens:
        # Condition de chevauchement : pas de séparation totale
        if not (st['end'] <= g_start or st['start'] >= g_end):
            matched_pos = st['pos']
            matched_ent = st['ent'] if st['ent'] else "—"
            break
    
    print(f"{gpt_str:15s} | ({g_start:2d},{g_end:2d})   | {matched_pos:8s} | {matched_ent:10s}")
