import os
import pandas as pd

from sklearn.model_selection import train_test_split
import datasets
from datasets import Dataset, DatasetDict

def tokenize_fn(examples):
    tokenized_inputs = tokenizer(examples['inputs'], examples['aspects'], padding='max_length', truncation=True, max_length=256)
    tokenized_inputs['label'] = examples['sentiment']
    return tokenized_inputs

def process(data_dir, tokenizer):

    task2_df = pd.read_csv(os.path.join(data_dir, 'corrected_sent_data.csv'))
    task2_df = task2_df[~(task2_df['aspects'] == 'No aspect')]
    task2_df.sentiment = task2_df.sentiment.apply(lambda x: x + 1).astype(int)

    train_df, val_df = train_test_split(task2_df, test_size=0.2, random_state=42)

    dataset_dict = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(train_df).remove_columns(['__index_level_0__']),
        'validation': datasets.Dataset.from_pandas(val_df).remove_columns(['__index_level_0__']),
        # 'test': datasets.Dataset.from_pandas(test_df)
    })

    tokenized_datasets =  dataset_dict.map(tokenize_fn, batched=True, remove_columns=['inputs', 'aspects', 'sentiment'])
    tokenized_datasets.set_format('torch')

    return tokenized_datasets


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=True)
    print(process('/workspaces/ABSAPT2024_Solutions/data', tokenizer))
