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
    f1 = f1_score.compute(predictions=predictions, references=labels)["f1"]
    r = recall.compute(predictions=predictions, references=labels)["recall"]
    p = precision.compute(predictions=predictions, references=labels)["precision"]
    
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc
    }