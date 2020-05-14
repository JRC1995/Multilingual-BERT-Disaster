import sys
sys.path.append("../")  # nopep8
import Lib.eval as eval
import Lib.utils as utils
from Models.BERT_Classifier import Classifier as BERT_Encoder
from Models.BERT_mixup_Classifier import Classifier as BERT_mixup_Encoder
from Models.BERT_mixup_agg_Classifier import Classifier as BERT_mixup_agg_Encoder
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
import logging
import copy
logging.disable(logging.CRITICAL)


parser = argparse.ArgumentParser(description='Model Name and stuff')
parser.add_argument('--model', type=str, default="BERT_mixup",
                    choices=["BOW_mean", "FastText", "BERT_word_mixup", "BERT_sent_mixup",
                             "CNN", "XML_CNN", "DenseCNN",
                             "BiLSTM",
                             "BERT", "BERT_mixup"])
parser.add_argument('--language', type=str, default="en", choices=["en", "all", "analysis"])
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
              'BERT_mixup': BERT_mixup_Encoder]}

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
    filename = '../Processed_Data/train_data_en.json'
elif language == "analysis":
    filename = '../Processed_Data/train_data_analysis.json'
else:
    filename = '../Processed_Data/train_data.json'

with open(filename) as file:

    data = json.load(file)

    train_texts = data["tweets"]
    train_labels = data["labels"]


if language == "en":
    filename = '../Processed_Data/val_data_en.json'
elif language == "analysis":
    filename = '../Processed_Data/val_data_analysis.json'
else:
    filename = '../Processed_Data/val_data.json'


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

    if train:

        model = model.train()

        if 'manifold_mixup' in config or 'word_mixup' in config or 'sent_mixup' in config:

            mixup_dict = {}

            if 'manifold_mixup' in config:
                layers = [i for i in range(13)]
                mixup_dict["layer_num"] = random.choice(layers)
                print(mixup_dict["layer_num"])
            elif 'word_mixup' in config:
                # print("hello")
                mixup_dict["layer_num"] = 0
            elif 'sent_mixup' in config:
                print("hello")
                mixup_dict["layer_num"] = 12

            alpha = config["mixup_alpha"]
            indices = [i for i in range(N)]
            random.shuffle(indices)
            lam = np.random.beta(alpha, alpha)

            mixup_dict["lam"] = lam
            mixup_dict["shuffled_indices"] = indices

            shuffled_label_mask = label_mask[indices]
            label_mask = lam*label_mask + (1-lam)*shuffled_label_mask

            shuffled_labels = labels[indices]
            labels = lam*labels + (1-lam)*shuffled_labels

            if 'n_fold' not in config:

                logits = model(text_ids, input_mask, mixup_dict)

            else:

                shuffled_indices_list = []
                indices_ = copy.deepcopy(indices)
                agg_labels = [labels]
                agg_label_masks = [label_mask]
                agg_label_weights = [label_weights_]

                for i in range(config['n_fold']-1):
                    random.shuffle(indices_)
                    shuffled_indices = copy.deepcopy(indices_)
                    shuffled_indices_list.append(shuffled_indices)
                    agg_labels.append(labels[shuffled_indices])
                    agg_label_masks.append(label_mask[shuffled_indices])
                    agg_label_weights.append(label_weights_)

                logits = model(text_ids, input_mask, mixup_dict, shuffled_indices_list)
                labels = T.cat(agg_labels, dim=-1)
                label_mask = T.cat(agg_label_masks, dim=-1)
                label_weights_ = T.cat(agg_label_weights, dim=-1)

        else:

            logits = model(text_ids, input_mask)

    else:

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

        model = model.eval()

        logits = model(text_ids, input_mask)

    # print(binary_logits.detach().cpu().numpy())
    if "n_fold" in config:
        predictions = logits[:, 0:len(idx2labels)].detach().cpu().numpy()
    else:
        predictions = logits.detach().cpu().numpy()

    predictions = np.where(predictions > 0.5, 1, 0).tolist()

    if loss_type == 'DSC':
        n_fold = config.get('n_fold', 1)
        loss = utils.DSC(model, n_fold, logits, labels, label_mask, label_weights_)

    else:
        loss = utils.multi_binary_cross_entropy(model,
                                                logits, labels,
                                                label_mask, label_weights_)

    T.cuda.empty_cache()

    return predictions, loss


val_batches_texts, val_batches_text_ids, val_batches_labels, \
    val_batches_mask, val_batches_label_masks = bnb.bucket_and_batch(
        val_texts, val_labels, labels2idx, vocab2idx, val_batch_size)

print("Validation batches loaded")

train_batches_texts, train_batches_text_ids, train_batches_labels, \
    train_batches_mask, train_batches_label_masks = bnb.bucket_and_batch(
        train_texts, train_labels, labels2idx, vocab2idx,  train_batch_size)

print("Train batches loaded")

display_step = 100
example_display_step = 500
patience = 5
meta_patience = 1
epochs = 100


time = 0

T.manual_seed(101)
random.seed(101)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False
np.random.seed(101)

while time < times:

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

    # checkpoint = T.load('../Model_Backup/tweet_BERT.pt')
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)

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

    optimizer = T.optim.AdamW([{'params': decay_parameters,
                                'lr': config["lr"], 'weight_decay': config["wd"]},
                               {'params': no_decay_parameters,
                                'lr': config["lr"], 'weight_decay': 0},
                               {'params': fine_tune_decay_parameters,
                                'lr': config["fine_tune_lr"], 'weight_decay': config["wd"]},
                               {'params': fine_tune_no_decay_parameters,
                                'lr': config["fine_tune_lr"], 'weight_decay': 0}],
                              lr=config["lr"])

    def lambda_(epoch): return (1 / 10)**epoch

    scheduler = T.optim.lr_scheduler.LambdaLR(optimizer, [lambda_]*4)

    load = 'n'  # input("\nLoad checkpoint? y/n: ")
    print("")

    if loss_type == 'DSC':
        checkpoint_path = "../Model_Backup/{}_{}_{}_DSC.pt".format(model_name, language, time)
    else:
        checkpoint_path = "../Model_Backup/{}_{}_{}.pt".format(model_name, language, time)

    if load.lower() == 'y':

        print('Loading pre-trained weights for the model...')

        checkpoint = T.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        past_epoch = checkpoint['past epoch']
        best_val_F1 = checkpoint['best F1']
        best_val_cost = checkpoint['best loss']
        impatience = checkpoint['impatience']
        meta_impatience = checkpoint['meta_impatience']
        time = checkpoint['time']

        print('\nRESTORATION COMPLETE\n')

    else:

        past_epoch = 0
        best_val_cost = math.inf
        best_val_F1 = -math.inf
        impatience = 0
        meta_impatience = 0

    print("\n\nTime:", time)

    for epoch in range(past_epoch, epochs):

        batches_indices = [i for i in range(0, len(train_batches_texts))]
        random.shuffle(batches_indices)

        total_train_loss = 0
        total_F1 = 0

        for i in range(len(train_batches_texts)):

            j = int(batches_indices[i])

            predictions, loss = predict(text_ids=train_batches_text_ids[j],
                                        labels=train_batches_labels[j],
                                        input_mask=train_batches_mask[j],
                                        label_mask=train_batches_label_masks[j],
                                        train=True)

            loss = loss/accu_step

            loss.backward()

            if (i+1) % accu_step == 0:
                # Update accumulated gradients
                T.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            labels = train_batches_labels[j].tolist()
            label_masks = train_batches_label_masks[j].tolist()

            prec, rec, acc = eval.multi_metrics(predictions, labels, label_masks, idx2labels)
            F1 = eval.compute_F1(prec, rec)

            cost = loss.item()

            if i % display_step == 0:

                print("Iter "+str(i)+", Cost = " +
                      "{:.3f}".format(cost)+", F1 = " +
                      "{:.3f}".format(F1)+", Accuracy = " +
                      "{:.3f}".format(acc))

            if i % example_display_step == 0:

                display(train_batches_texts[j],
                        predictions, labels)

        print("\n\n")

        total_val_cost = 0
        batch_labels = []
        batch_binary_labels = []
        batch_predictions = []
        batch_label_masks = []

        for i in range(0, len(val_batches_texts)):

            if i % display_step == 0:
                print("Validating Batch {}".format(i+1))

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

            if i % example_display_step == 0:

                display(val_batches_texts[i],
                        predictions, labels)

        prec, rec, acc = eval.multi_metrics(batch_predictions,
                                            batch_labels,
                                            batch_label_masks,
                                            idx2labels,
                                            verbose=False)
        val_F1 = eval.compute_F1(prec, rec)

        val_len = len(val_batches_texts)

        avg_val_cost = total_val_cost/val_len

        print("\n\nVALIDATION\n\n")

        print("Epoch "+str(epoch)+":, Cost = " +
              "{:.3f}".format(avg_val_cost)+", F1 = " +
              "{:.3f}".format(val_F1)+", Accuracy = " +
              "{:.3f}".format(acc))

        flag = 0
        impatience += 1

        if avg_val_cost < best_val_cost:

            impatience = 0

            best_val_cost = avg_val_cost

            # flag = 1

        if val_F1 >= best_val_F1:

            impatience = 0

            best_val_F1 = val_F1

            flag = 1

        if flag == 1:

            T.save({
                'past epoch': epoch+1,
                'best loss': best_val_cost,
                'best F1': best_val_F1,
                'impatience': impatience,
                'meta_impatience': meta_impatience,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'time': time
            }, checkpoint_path)

            print("Checkpoint created!")

        print("\n")

        if impatience > patience:
            # scheduler.step()
            # meta_impatience += 1
            # impatience = 0
            # if meta_impatience > meta_patience:
            break

    time += 1
