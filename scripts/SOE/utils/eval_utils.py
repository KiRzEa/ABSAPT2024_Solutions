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
    mac_p = precision.compute(predictions=predictions, references=labels, average='macro')["precision"]
    mic_p = precision.compute(predictions=predictions, references=labels, average='micro')["precision"]
    weighted_p = precision.compute(predictions=predictions, references=labels, average='weighted')["precision"]
    mac_r = recall.compute(predictions=predictions, references=labels, average='macro')["recall"]
    mic_r = recall.compute(predictions=predictions, references=labels, average='micro')["recall"]
    weighted_r = recall.compute(predictions=predictions, references=labels, average='weighted')["recall"]

    return {
        "MacroF1": mac_f1,
        "MacroPrecision": mac_p,
        "MacroRecall": mac_r,
        "MicroF1": mic_f1,
        "MicroPrecision": mic_p,
        "MicroRecall": mic_r,
        "WeightedF1": weighted_f1,
        "WeightedPrecision": weighted_p,
        "WeightedRecall": weighted_r,
        "Accuracy": acc,
        "BalancedAccuracy": bal_acc
    }