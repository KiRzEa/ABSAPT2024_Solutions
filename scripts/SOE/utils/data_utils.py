import os
import pandas as pd

import datasets

def tokenize_fn(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(examples['inputs'], examples['aspects'], padding='max_length', truncation=True, max_length=max_length)
    try:
        tokenized_inputs['label'] = examples['sentiment']
    except:
        return tokenized_inputs
    return tokenized_inputs

def process(data_dir, tokenizer, level):

    train_df = pd.read_csv(os.path.join(data_dir, f'task2/{level}/train.csv'), delimiter=';')
    val_df = pd.read_csv(os.path.join(data_dir, f'task2/{level}/val.csv'), delimiter=';')
    test_df = pd.read_csv(os.path.join(data_dir, f'task2/{level}/test.csv'), delimiter=';')
    train_df = train_df[~(train_df['aspects'] == 'No aspect')]
    val_df = val_df[~(val_df['aspects'] == 'No aspect')]
    train_df.sentiment = train_df.sentiment.apply(lambda x: x + 1).astype(int)
    val_df.sentiment = val_df.sentiment.apply(lambda x: x + 1).astype(int)

    dataset_dict = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(train_df),
        'validation': datasets.Dataset.from_pandas(val_df),
        'test': datasets.Dataset.from_pandas(test_df),
    })

    tokenized_datasets =  dataset_dict.map(tokenize_fn, fn_kwargs={'tokenizer': tokenizer, 'max_length': 512 if level == 'document' else 256}, batched=True)
    tokenized_datasets.set_format('torch')

    return tokenized_datasets


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=True)
    print(process('/workspaces/ABSAPT2024_Solutions/data', tokenizer, level='sentence'))
