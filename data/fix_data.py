import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

def tokenize(text, tokenizer):
    tokenized = tokenizer(text)
    return ' '.join(tokenizer.convert_ids_to_tokens(tokenized['input_ids'])[1:-1])

def check_pattern(words, aspect, i, start_position, end_position):
    return ' '.join(words[i:i+len(aspect.split())]) == aspect

cnt = 0

def process_label(example, tokenizer):
    text = tokenize(example.texto.replace('´', '\''), tokenizer)
    aspect_1 = tokenize(example.aspect.lower(), tokenizer) # lower
    aspect_2 = tokenize(example.aspect.capitalize(), tokenizer) # capital
    aspect_3 = tokenize(example.aspect.upper(), tokenizer) # upper
    aspect_4 = tokenize(f'{example.aspect.lower()}s', tokenizer)
    aspect_5 = tokenize(f'{example.aspect.lower()}es', tokenizer)
    aspect_6 = tokenize(f'{example.aspect.capitalize()}s', tokenizer)
    aspect_7 = tokenize(f'{example.aspect.capitalize()}es', tokenizer)
    aspect_8 = tokenize(f'{example.aspect.upper()}S', tokenizer)
    aspect_9 = tokenize(f'{example.aspect.upper()}ES', tokenizer)
    aspect_10 = tokenize(example.aspect.title().replace(" Da ",  " da ").replace(" De ", " de "), tokenizer)

    words = text.split()
    for i in range(len(words)):
        if check_pattern(words, aspect_1, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_1.split())
            break
        elif check_pattern(words, aspect_2, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_2.split())
            break
        elif check_pattern(words, aspect_3, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_3.split())
            break
        elif check_pattern(words, aspect_4, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_4.split())
            break
        elif check_pattern(words, aspect_5, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_5.split())
            break
        elif check_pattern(words, aspect_6, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_6.split())
            break
        elif check_pattern(words, aspect_7, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_7.split())
            break
        elif check_pattern(words, aspect_8, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_8.split())
            break
        elif check_pattern(words, aspect_9, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_9.split())
            break
        elif check_pattern(words, aspect_10, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_10.split())
            break
 
    is_found = True
    try:
      aspect_span = (aspect_start_index, aspect_end_index)
    except:
      aspect_span = ""
      is_found = False
      print(aspect_1)
      print(text)
      print(example.texto)
      print('=' * 50)
    try:
      return text, f"{aspect_span} {example.polarity}", is_found
    except:
      return text, f"{aspect_span}", is_found


train_data = pd.read_csv('train2024.csv')

for _, row in train_data.iterrows():
  text, label, is_found = process_label(row, tokenizer)
  row['label'] = label
  row['tokenized_text'] = text
  if not is_found:
    cnt += 1

print(f"{cnt} Error(s)")

cnt = 0
test_data = pd.read_csv('task2_test.csv')

for _, row in test_data.iterrows():
  text, label, is_found = process_label(row, tokenizer)
  row['label'] = label
  row['tokenized_text'] = text
  if not is_found:
    cnt += 1

print(f"{cnt} Error(s)")

