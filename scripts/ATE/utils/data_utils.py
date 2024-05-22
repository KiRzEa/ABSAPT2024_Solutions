import os
import pandas as pd

import nltk
nltk.download('punkt')
from nltk.tokenize.treebank import TreebankWordTokenizer

import datasets
from datasets import Dataset, DatasetDict

def convert_to_bio(df):
    data = []
    for i, row in df.iterrows():
        aspects_span = [[i, j, 0] for i, j in zip(row['start_position'], row['end_position'])]
        tokens = []
        ner_tags = []
        span_generator = TreebankWordTokenizer().span_tokenize(row['inputs'])
        for span in span_generator:
            tokens.append(row['inputs'][span[0]:span[1]])
            is_aspect = False
            aspect_data = None
            for aspect_span in aspects_span:
                if is_span_a_subset(span, aspect_span):
                    is_aspect = True
                    aspect_data = aspect_span
            if is_aspect:
                label = 'ASPECT'

                if aspect_data[-1] == 0:
                    ner_tags.append('B-'+label)
                    aspect_data[-1] = aspect_data[-1] + 1
                else:
                    ner_tags.append('I-'+label)
            else:
                ner_tags.append('O')
        data.append({'id': i, 'tokens': tokens, 'ner_tags': ner_tags})
    return data

def tokenize(example):
    tokens = []
    span_generator = TreebankWordTokenizer().span_tokenize(example['inputs'].replace('`', '\''))
    for span in span_generator:
        tokens.append(example['inputs'][span[0]:span[1]])
    return tokens

def is_span_a_subset(span, aspect_span):
    if span[0] >= aspect_span[1]:
        return False
    elif span[1] < aspect_span[0]:
        return False
    else:
        return True

def tokenize_and_align_labels(dataset_unaligned, tokenizer, max_length, label_all_tokens=False):
    tokenized_inputs = tokenizer(dataset_unaligned["tokens"], truncation=True, is_split_into_words=True, max_length=512)
    labels = []
    for i, label in enumerate(dataset_unaligned[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None: #special tokens
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            else: # subwords
                label_ids.append(1 if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_fn(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, max_length=512)
    pseudo_labels = []
    for i, _ in enumerate(examples['tokens']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(-1)

            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        pseudo_labels.append(label_ids)
    tokenized_inputs["pseudo_labels"] = pseudo_labels
    return tokenized_inputs

def sent_process(data_dir, tokenizer):
    raise NotImplementedError

def process(data_dir, tokenizer, level):

    ate_train_df = pd.read_csv(os.path.join(data_dir, f'task1/{level}/train.csv'), delimiter=';')
    ate_dev_df = pd.read_csv(os.path.join(data_dir, f'task1/{level}/val.csv'), delimiter=';')
    ate_test_df = pd.read_csv(os.path.join(data_dir, f'task1/{level}/test.csv'), delimiter=';')

    ate_test_data = ate_test_df.copy()
    ate_test_data['tokens'] = ate_test_data.apply(tokenize, axis=1)

    # ate_train_data = ate_train_df.groupby('inputs').agg(list).reset_index()
    # ate_dev_data = ate_dev_df.groupby('inputs').agg(list).reset_index()
    ate_train_data = ate_train_df[['aspects', 'start_position', 'end_position']].map(eval)
    ate_dev_data = ate_dev_df[['aspects', 'start_position', 'end_position']].map(eval)
    
    train_ds = Dataset.from_pandas(pd.DataFrame(convert_to_bio(ate_train_df)))
    dev_ds = Dataset.from_pandas(pd.DataFrame(convert_to_bio(ate_dev_df)))
    test_ds = Dataset.from_pandas(ate_test_data[['id', 'tokens']])

    label_list = sorted(list(set(tag for doc in train_ds['ner_tags'] for tag in doc)))

    train_features = datasets.Features(
        {
        'id': datasets.Value('int32'),
        'tokens': datasets.Sequence(datasets.Value('string')),
        'ner_tags': datasets.Sequence(
            datasets.features.ClassLabel(names=label_list)
            )
        }
    )

    test_features = datasets.Features(
        {
        'id': datasets.Value('int32'),
        'tokens': datasets.Sequence(datasets.Value('string'))
        }
    )
    
    train_ds = train_ds.map(train_features.encode_example, features=train_features)
    dev_ds = dev_ds.map(train_features.encode_example, features=train_features)
    test_ds = test_ds.map(test_features.encode_example, features=test_features)

    tokenized_train = train_ds.map(tokenize_and_align_labels, fn_kwargs={'tokenizer': tokenizer, 'max_length': 512 if level == 'document' else 256}, batched=True)
    tokenized_dev = dev_ds.map(tokenize_and_align_labels, fn_kwargs={'tokenizer': tokenizer, 'max_length': 512 if level == 'document' else 256}, batched=True)
    tokenized_test = test_ds.map(tokenize_fn, fn_kwargs={'tokenizer': tokenizer, 'max_length': 512 if level == 'document' else 256}, batched=True)

    tokenized_datasets = DatasetDict({
        'train': tokenized_train,
        'validation': tokenized_dev,
        'test': tokenized_test
    })

    return tokenized_datasets


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=True)
    print(process('/workspaces/ABSAPT2024_Solutions/data', tokenizer, 'sentence'))
