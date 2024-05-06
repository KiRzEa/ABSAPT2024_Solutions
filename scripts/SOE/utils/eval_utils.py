import evaluate
import numpy as np
from sklearn.metrics import *

accuracy = evaluate.load("accuracy")
f1_score = evaluate.load("f1")
recall = evaluate.load("recall")
precision = evaluate.load("precision")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    bal_acc = balanced_accuracy_score(labels, predictions)
    acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    mac_f1 = f1_score.compute(predictions=predictions, references=labels, average='macro')["f1"]
    mic_f1 = f1_score.compute(predictions=predictions, references=labels, average='micro')["f1"]
    weighted_f1 = f1_score.compute(predictions=predictions, references=labels, average='weighted')["f1"]

    return {
        "Macro F1-score": mac_f1,
        "Micro F1-score": mic_f1,
        "Weighted F1-score": weighted_f1,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc
    }