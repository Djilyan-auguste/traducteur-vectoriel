from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = 'Marie Curie was born in Paris on November 7, 1867. She was a physicist and chemist.'
encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

print('=== GPT-2 TOKENS ===')
for i, (token_id, offset) in enumerate(zip(encoded['input_ids'], encoded['offset_mapping'])):
    token_str = tokenizer.decode([token_id])
    print(f'Token {i:2d}: {token_str:12s} | offset={offset} | id={token_id}')

print(f'\nTotal tokens: {len(encoded["input_ids"])}')
