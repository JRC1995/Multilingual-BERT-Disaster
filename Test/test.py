import sys
sys.path.append("../")  # nopep8
import Lib.eval as eval
import Lib.utils as utils
from Models.BERT_Classifier import Classifier as BERT_Encoder
from Models.BERT_mixup_Classifier import Classifier as BERT_mixup_Encoder
from Models.BERT_mixup_agg_Classifier import Classifier as BERT_mixup_agg_Encoder
from Models.BERT_mixup_MTL_Classifier import Classifier as BERT_mixup_MTL_Encoder
from Models.TweetBERT_mixup_Classifier import Classifier as TweetBERT_mixup_Encoder
from Models.BiLSTM_Classifier import Classifier as BiLSTM_Encoder
from Models.DenseCNN_Classifier import Classifier as DenseCNN_Encoder
from Models.XML_CNN_Classifier import Classifier as XML_CNN_Encoder
from Models.CNN_Classifier import Classifier as CNN_Encoder
from Models.FastText_Classifier import Classifier as FastText_Encoder
from Models.BOW_mean_Classifier import Classifier as BOW_mean_Encoder
from DataLoader.bucket_and_batch import bucket_and_batch
from DataLoader.bucket_and_batch_bert import bucket_and_batch as bucket_and_batch_bert
import pickle
from fasttext import load_model
from transformers import *
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch as T
import argparse
import json
import random
import string
import numpy as np
import math
import copy
import logging
logging.disable(logging.CRITICAL)


parser = argparse.ArgumentParser(description='Model Name and stuff')
parser.add_argument('--model', type=str, default="BERT_sent_mixup",
                    choices=["BOW_mean", "FastText", "BERT_word_mixup", "BERT_sent_mixup",
                             "CNN", "XML_CNN", "DenseCNN",
                             "BiLSTM",
                             "BERT", "BERT_mixup"])
parser.add_argument('--language', type=str, default="all", choices=["en", "all", "analysis"])
parser.add_argument('--loss', type=str, default="CE", choices=["DSC", "CE"])
parser.add_argument('--times', type=int, default=5)
flags = parser.parse_args()

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)

model_name = flags.model
language = flags.language
times = flags.times
loss_type = flags.loss

if language != "en" and language != "analysis":
    language = "all"

print("\n\nTraining Model: {}\n\n".format(model_name))

model_dict = {'BOW_mean': BOW_mean_Encoder,
              'FastText': FastText_Encoder,
              'CNN': CNN_Encoder,
              'XML_CNN': XML_CNN_Encoder,
              'DenseCNN': DenseCNN_Encoder,
              'BiLSTM': BiLSTM_Encoder,
              'BERT': BERT_Encoder,
              'BERT_word_mixup': BERT_mixup_Encoder,
              'BERT_sent_mixup': BERT_mixup_Encoder,
              'BERT_mixup': BERT_mixup_Encoder}


Encoder = model_dict.get(model_name, BiLSTM_Encoder)


random.seed(101)

if 'bert' in model_name.lower():
    bnb = bucket_and_batch_bert()
else:
    bnb = bucket_and_batch()

with open("../Configs/{}_config.json".format(model_name), "r") as file:
    config = json.load(file)

val_batch_size = config["val_batch_size"]
train_batch_size = config["train_batch_size"]
total_train_batch_size = config["total_train_batch_size"]
accu_step = total_train_batch_size//train_batch_size
max_grad_norm = config["max_grad_norm"]


if language == "en":
    filename = '../Processed_Data/test_data_en.json'
elif language == "analysis":
    filename = '../Processed_Data/test_data_analysis.json'
else:
    filename = '../Processed_Data/test_data.json'


with open(filename) as file:

    data = json.load(file)

    val_texts = data["tweets"]
    val_labels = data["labels"]


with open('../Processed_Data/label_info.json') as file:
    label_weights = json.load(file)

labels2idx = {'casaulties and damage': 0,
              'caution and advice': 1,
              'donation': 2,
              'informative': 3}

idx2labels = {v: k for k, v in labels2idx.items()}

with open('../Processed_Data/vocab.pkl', 'rb') as file:
    data = pickle.load(file)

vocab2idx = data['vocab2idx']

if 'fasttext' in model_name.lower():
    embeddings = data['ft_embeddings']
elif 'bert' in model_name.lower():
    pass
else:
    embeddings = data['ft_embeddings']

if 'bert' in model_name.lower():
    model = Encoder(classes_num=len(labels2idx),
                    config=config,
                    device=device)
else:
    model = Encoder(embeddings=embeddings,
                    pad_idx=vocab2idx['<PAD>'],
                    classes_num=len(labels2idx),
                    config=config,
                    device=device)


model = model.to(device)

decay_parameters = []
fine_tune_decay_parameters = []
no_decay_parameters = []
fine_tune_no_decay_parameters = []
allowed_layers = [11, 10, 9, 8]

if 'bert' in model_name.lower():

    for name, param in model.named_parameters():
        if "bert" not in name.lower():
            if "bias" in name.lower():
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)
        else:
            if 'full' in model_name.lower():
                if "bias" or '_embedding' in name.lower():
                    fine_tune_no_decay_parameters.append(param)
                else:
                    fine_tune_decay_parameters.append(param)
            else:

                flag = 0
                for layer_num in allowed_layers:
                    layer_num = str(layer_num)
                    if ".{}.".format(layer_num) in name:
                        if 'bias' in name.lower():
                            fine_tune_no_decay_parameters.append(param)
                        else:
                            fine_tune_decay_parameters.append(param)
                        flag = 1
                        break
                """
                if '_embedding' in name.lower() and flag == 0:
                    fine_tune_no_decay_parameters.append(param)
                    flag = 1
                """

                if 'pooler' in name.lower() and flag == 0:
                    fine_tune_decay_parameters.append(param)
                    flag = 1

                if flag == 0:
                    param.requires_grad = False

else:

    for name, param in model.named_parameters():
        if "embedding" in name.lower():
            fine_tune_no_decay_parameters.append(param)
        else:
            if 'bias' in name.lower():
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)

parameters = decay_parameters+no_decay_parameters + \
    fine_tune_decay_parameters+fine_tune_no_decay_parameters
parameter_count = sum(p.numel() for p in parameters if p.requires_grad is True)

print("Parameters:\n\n")
for name, param in model.named_parameters():
    if param.requires_grad is True:
        print(name, param.size())
print("\n")
print("Parameter Count: ", parameter_count)
print("\n")


def display(texts, predictions, labels):

    global idx2labels
    global labels2idx

    N = len(texts)
    j = random.choice(np.arange(N).tolist())

    display_text = texts[j]
    display_prediction_idx = [i for i, v in enumerate(predictions[j]) if v == 1]
    display_gold_idx = [i for i, v in enumerate(labels[j]) if v == 1]
    display_prediction = ", ".join([idx2labels[i] for i in display_prediction_idx])
    display_gold = ", ".join([idx2labels[i] for i in display_gold_idx])

    if len(display_gold_idx) == 0:
        display_gold = "not informative"

    if len(display_prediction_idx) == 0:
        display_prediction = "not informative"

    print("\n\nExample Prediction\n")
    print("Text: {}\n".format(display_text))
    print("PREDICTION: {} | GOLD: {}\n".format(display_prediction, display_gold))


def predict(text_ids, labels, input_mask, label_mask, train=True):

    global model
    global label_weights
    global idx2labels
    global model_name

    with T.no_grad():

        text_ids = T.tensor(text_ids).long().to(device)
        labels = T.tensor(labels).long().to(device)
        input_mask = T.tensor(input_mask).float().to(device)
        label_mask = T.tensor(label_mask).float().to(device)
        label_weights_ = [label_weights[idx2labels[i]] for i in range(len(idx2labels))]
        label_weights_ = T.tensor(np.asarray(label_weights_, np.float32)).to(device)
        label_weights_ = label_weights_.view(1, -1)
        N = text_ids.size()[0]
        label_weights_ = T.repeat_interleave(label_weights_, repeats=N, dim=0)

    model = model.eval()

    if 'n_fold' in config:
        agg_labels = [labels]
        agg_label_masks = [label_mask]
        agg_label_weights = [label_weights_]

        for i in range(config['n_fold']-1):
            agg_labels.append(labels)
            agg_label_masks.append(label_mask)
            agg_label_weights.append(label_weights_)

        labels = T.cat(agg_labels, dim=-1)
        label_mask = T.cat(agg_label_masks, dim=-1)
        label_weights_ = T.cat(agg_label_weights, dim=-1)

    if 'mtl' in model_name.lower():
        logits, _ = model(text_ids, input_mask)
    else:
        logits = model(text_ids, input_mask)

    # print(binary_logits.detach().cpu().numpy())

    if "n_fold" in config:
        n_fold = config['n_fold']
        predictions = logits[:, 0:len(idx2labels)].mean(dim=1).detach().cpu().numpy()
    else:
        predictions = logits.detach().cpu().numpy()
    predictions = np.where(predictions > 0.5, 1, 0).tolist()

    loss = utils.multi_binary_cross_entropy(model,
                                            logits, labels,
                                            label_mask, label_weights_)

    T.cuda.empty_cache()

    return predictions, loss


for disaster in val_texts:

    print("\n\n{}\n\n".format(disaster))

    val_batches_texts, val_batches_text_ids, val_batches_labels, \
        val_batches_mask, val_batches_label_masks = bnb.bucket_and_batch(
            val_texts[disaster], val_labels[disaster], labels2idx, vocab2idx, val_batch_size)

    F1s = []
    accs = []

    for time in range(times):

        checkpoint = T.load(
            "../Model_Backup/{}_{}_{}.pt".format(model_name, language, time))  # model_name,
        model.load_state_dict(checkpoint['model_state_dict'])

        #print("Test batches loaded")

        display_step = 100
        example_display_step = 500
        patience = 5
        meta_patience = 1

        total_val_cost = 0
        batch_labels = []
        batch_binary_labels = []
        batch_predictions = []
        batch_label_masks = []

        for i in range(0, len(val_batches_texts)):

            if i % display_step == 0:
                pass
                #print("Testing Batch {}".format(i+1))

            with T.no_grad():

                predictions, loss = predict(text_ids=val_batches_text_ids[i],
                                            labels=val_batches_labels[i],
                                            input_mask=val_batches_mask[i],
                                            label_mask=val_batches_label_masks[i],
                                            train=False)

                cost = loss.item()

                total_val_cost += cost

                labels = val_batches_labels[i].tolist()
                label_masks = val_batches_label_masks[i].tolist()

                batch_labels += labels
                batch_predictions += predictions
                batch_label_masks += label_masks

            """

            if i % example_display_step == 0:

                display(val_batches_texts[i],
                        predictions, labels)
            """

        prec, rec, acc = eval.multi_micro_metrics(batch_predictions,
                                                  batch_labels,
                                                  batch_label_masks,
                                                  idx2labels,
                                                  verbose=False)
        val_F1 = eval.compute_F1(prec, rec)

        val_len = len(val_batches_texts)

        avg_val_cost = total_val_cost/val_len

        F1s.append(val_F1*100)
        accs.append(acc*100)

    mean_F1 = sum(F1s)/times
    std_F1 = np.std(F1s)
    mean_acc = sum(accs)/times
    std_acc = np.std(accs)

    # print(F1s)

    print("Disaster: {}, F1: {:.3f}+-{:.3f}, Accuracy: {:.3f}+-{:.3f}".format(disaster,
                                                                              mean_F1, std_F1, mean_acc, std_acc))
