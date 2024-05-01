import pandas as pd
import numpy as np
from evaluate import load_metric

metric = load_metric('seqeval', trust_remote_code=True)
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
        
        return predictions, labels, pd.DataFrame(results)