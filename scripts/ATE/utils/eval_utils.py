import pandas as pd
import numpy as np
import evaluate

metric = evaluate.load('seqeval', trust_remote_code=True)
label_list = ['B-ASPECT', 'I-ASPECT', 'O']

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def concat_lists(series):
    concatenated_list = []
    for lst in series:
        concatenated_list.extend(lst)
    return set(concatenated_list)

def extract_aspect(tokens, ner_tags):
    aspects = []
    aspect_tokens = []
    for idx, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag == 'B-ASPECT':
          aspect_tokens = [token]
        elif tag == 'I-ASPECT':
          aspect_tokens.append(token)
        elif tag == 'O':
          if len(aspect_tokens) > 0:
            aspects.append(' '.join(aspect_tokens))
            aspect_tokens = []

    return aspects

def predict(trainer, ds, inference=False):
    logits, labels, _ = trainer.predict(ds)
    predictions = np.argmax(logits, axis=2)

    if inference:
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, pseudo_label) if l != -100]
            for prediction, pseudo_label in zip(predictions, ds['pseudo_labels'])
        ]
        return true_predictions
    else:
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        return true_predictions, true_labels, pd.DataFrame(results)

def eval_ate(preds, golds):
    tp = .0
    fp = .0
    fn = .0
    n_total = 0
    for pred, gold in zip(preds, golds):   
        n_total += len(gold)
        for aspect in gold:
            if aspect in pred:
                tp += 1
            else:
                fn += 1
        for aspect in pred:
            if aspect not in gold:
                fp += 1

    precision = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    recall = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
    acc = tp / n_total
    print(f"tp: {tp}, fp: {fp}, fn: {fn}")
    print(f"p: {precision}, r: {recall}, f1: {f1}, acc: {acc}")
    return {'precision': precision, 'recall': recall, 'f1': f1, "acc": acc}