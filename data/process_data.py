import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import argparse
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

from transformers import AutoTokenizer

def tokenize(text, tokenizer):
    tokenized = tokenizer(text)
    return ' '.join(tokenizer.convert_ids_to_tokens(tokenized['input_ids'])[1:-1])

def check_pattern(words, aspect, i, start_position, end_position):
    return ' '.join(words[i:i+len(aspect.split())]) == aspect

cnt = 0

def process_label(example, tokenizer):
    text = tokenize(example.texto.replace('´', '\''), tokenizer)
    aspect_1 = tokenize(example.aspect.lower(), tokenizer) # lower
    aspect_2 = tokenize(f'{example.aspect.lower()}s', tokenizer)
    aspect_3 = tokenize(f'{example.aspect.lower()}es', tokenizer)

    words = text.split()
    for i in range(len(words)):
        if check_pattern(words, aspect_1, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_1.split())
            break
        if check_pattern(words, aspect_2, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_2.split())
            break
        if check_pattern(words, aspect_3, i, example.start_position, example.end_position):
            aspect_start_index = i
            aspect_end_index = i + len(aspect_3.split())
            break
 
    aspect_span = (aspect_start_index, aspect_end_index)

    return f"{aspect_span}"

def preprocess_data(data_dir, output_dir, tokenizer):

    train_data = pd.read_csv(os.path.join(data_dir, 'train2024.csv'), delimiter=';')
    test1 = pd.read_csv(os.path.join(data_dir, 'task1_test.csv'), delimiter=';')
    test2 = pd.read_csv(os.path.join(data_dir, 'task2_test.csv'), delimiter=';')

    train_data['span'] = train_data.apply(lambda x: process_label(x, tokenizer), axis=1)
    test2['span'] = test2.apply(lambda x: process_label(x, tokenizer), axis=1)

    train_data = train_data.groupby('texto').agg({
        'aspect': list,
        'span': list,
        'polarity': list,
    }).reset_index()
    test2 = test2.groupby('texto').agg({
        'aspect': list,
        'span': list,
    }).reset_index()

    train_data['tokens'] = train_data.apply(lambda x: tokenize(x.texto, tokenizer).split(), axis=1)
    test1['tokens'] = test1.apply(lambda x: tokenize(x.texto, tokenizer), axis=1)
    test2['tokens'] = test2.apply(lambda x: tokenize(x.texto, tokenizer, axis=1))

    train_data[['tokens', 'aspect', 'span', 'polarity']].to_csv(os.path.join(output_dir, 'tokenized_data.csv'), index=False)
    test1[['tokens']].to_csv(os.path.join(output_dir, 'tokenized_test1.csv'), index=False)
    test2[['tokens']].to_csv(os.path.join(output_dir, 'tokenized_test2.csv'), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        required=True)
    parser.add_argument('--model_name',
                        type=str,
                        required=True)
    parser.add_argument('--output_dir',
                        type=str,
                        required=True)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preprocess_data(args.data_dir, args.output_dir, tokenizer)

if __name__ == '__main__':
    main()
