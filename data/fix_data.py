import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

def tokenize(text, tokenizer):
    return ' '.join(tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids'])[1:-1])

def process_label(example, tokenizer):
    text = tokenize(example.texto, tokenizer)
    aspect = tokenize(example.aspect, tokenizer)
    words = text.split()
    for i in range(len(words)):
        if ' '.join(words[i:i+len(aspect.split())]).lower() == aspect or ' '.join(words[i:i+len(aspect.split())]) == aspect:
            aspect_start_index = i
            aspect_end_index = i + len(aspect.split())
            break
    try:
      aspect_span = f"{aspect_start_index},{aspect_end_index}"
    except:
      print(row.aspect)
      print(row.texto[row.start_position:row.end_position])
      print(text)
      print(aspect)
    return text, f"{aspect_span} {example.polarity}"

train_data = pd.read_csv('train2024.csv', delimiter=';')

for _, row in tqdm(train_data.iterrows()):
  text, label = process_label(row, tokenizer)
  row['label'] = label
  row['tokenized_text'] = text
