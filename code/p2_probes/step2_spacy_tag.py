import spacy

nlp = spacy.load("en_core_web_sm")
text = 'Marie Curie was born in Paris on November 7, 1867. She was a physicist and chemist.'
doc = nlp(text)

print('=== SPACY TAGS ===')
for token in doc:
    print(f'Token: {token.text:12s} | POS: {token.pos_:8s} | Entity: {token.ent_type_ if token.ent_type_ else "—"}')

print('\n=== ENTITIES DETECTED ===')
for ent in doc.ents:
    print(f'Entity: {ent.text:20s} | Label: {ent.label_}')
