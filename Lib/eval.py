import numpy as np


def multi_micro_metrics(predictions, labels, label_masks, idx2labels, verbose=False):

    total_precision = 0
    total_recall = 0
    total_accuracy = 0

    tp = 0
    pred_len = 0
    gold_len = 0
    correct = 0
    total = 0

    for id in idx2labels:

        for prediction, label, label_mask in zip(predictions, labels, label_masks):
            if label_mask[id] == 1:
                if prediction[id] == label[id]:
                    correct += 1
                    if prediction[id] == 1:
                        tp += 1
                if prediction[id] == 1:
                    pred_len += 1
                if label[id] == 1:
                    gold_len += 1
                total += 1

    precision = tp/pred_len if pred_len > 0 else 0
    recall = tp/gold_len if gold_len > 0 else 0
    accuracy = correct/total if total > 0 else 0

    if verbose:
        print("\n")
        print("precision", precision)
        print("recall", recall)
        print("\n")

    return precision, recall, accuracy


def multi_metrics(predictions, labels, label_masks, idx2labels, verbose=False):

    total_precision = 0
    total_recall = 0
    total_accuracy = 0

    for id in idx2labels:
        tp = 0
        correct = 0
        pred_len = 0
        gold_len = 0
        total = 0
        for prediction, label, label_mask in zip(predictions, labels, label_masks):
            if label_mask[id] == 1:
                if prediction[id] == label[id]:
                    correct += 1
                    if prediction[id] == 1:
                        tp += 1
                if prediction[id] == 1:
                    pred_len += 1
                if label[id] == 1:
                    gold_len += 1
                total += 1
        precision = tp/pred_len if pred_len > 0 else 0
        recall = tp/gold_len if gold_len > 0 else 0
        accuracy = correct/total if total > 0 else 0

        if verbose:
            print("\n")
            print("Label", idx2labels[id])
            print("pred_len", pred_len)
            print("gold_len", gold_len)
            print("precision", precision)
            print("recall", recall)
            F1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
            print("F1", F1)

        total_precision += precision
        total_recall += recall
        total_accuracy += accuracy

    precision = total_precision/len(idx2labels)
    recall = total_recall/len(idx2labels)
    accuracy = total_accuracy/len(idx2labels)

    if verbose:
        print("\n")
        print("avg precision", precision)
        print("avg recall", recall)
        print("\n")

    return precision, recall, accuracy


def compute_F1(precision, recall):
    F1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    return F1
