import numpy as np
from transformers import TFBertModel, BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("indolem/indobert-base-uncased")

def tokenize(data, max_len):
    input_ids = []
    attention_masks = []
    for text in data:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])
    
    return np.array(input_ids), np.array(attention_masks)