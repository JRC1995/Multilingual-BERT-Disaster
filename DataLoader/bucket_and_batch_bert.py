import numpy as np
import random
import re
import pickle
import os
import logging
logging.basicConfig(level=logging.CRITICAL)
from transformers import *
import copy


class bucket_and_batch:

    def bucket_and_batch(self, texts, labels, labels2idx, vocab2idx, batch_size):

        idx2labels = {v: k for k, v in labels2idx.items()}
        original_texts = copy.deepcopy(texts)

        tokenizer = BertTokenizer.from_pretrained('../Embeddings/Pre_trained_BERT/')

        text_ids = [tokenizer.encode(text, add_special_tokens=True,
                                     max_length=300) for text in texts]
        PAD = tokenizer.encode([tokenizer.pad_token], add_special_tokens=False)[0]
        true_seq_lens = [len(text_id) for text_id in text_ids]

        # sorted in descending order after flip
        sorted_idx = np.flip(np.argsort(true_seq_lens), 0).tolist()

        sorted_texts = [original_texts[i] for i in sorted_idx]
        sorted_text_ids = [text_ids[i] for i in sorted_idx]
        sorted_labels = [labels[i] for i in sorted_idx]

        print("Sample size: ", len(sorted_texts))

        i = 0
        batches_texts = []
        batches_text_ids = []
        batches_labels = []
        batches_mask = []
        batches_label_masks = []
        batches_weights = []
        count = 0

        zeros_label = [0 for _ in range(len(labels2idx))]
        ones_label = [1 for _ in range(len(labels2idx))]

        informative_index = labels2idx['informative']

        while i < len(sorted_texts):

            if i+batch_size > len(sorted_texts):
                batch_size = len(sorted_texts)-i

            batch_texts = []
            batch_text_ids = []
            batch_labels = []
            batch_mask = []
            batch_label_masks = []
            batch_weights = []

            max_len = len(sorted_text_ids[i])

            for j in range(i, i + batch_size):

                text = sorted_texts[j]
                text_id = sorted_text_ids[j]
                attention_mask = [1 for k in range(len(text_id))]
                label = copy.deepcopy(zeros_label)

                if sorted_labels[j] == 'not informative':
                    label_mask = copy.deepcopy(ones_label)
                else:
                    if sorted_labels[j] == 'unlabeled':
                        label[informative_index] = 1
                        label_mask = copy.deepcopy(zeros_label)
                        label_mask[informative_index] = 1
                    else:
                        label_id = labels2idx[sorted_labels[j]]
                        label_mask = copy.deepcopy(ones_label)
                        label[label_id] = 1
                        label[informative_index] = 1

                while len(text_id) < max_len:
                    text_id.append(PAD)
                    attention_mask.append(0)

                # print(text)
                # print(label_mask)
                # print("multi:", idx2labels[label])
                # print("binary:", binary_idx2labels[binary_label])

                batch_texts.append(text)
                batch_text_ids.append(text_id)
                batch_labels.append(label)
                batch_mask.append(attention_mask)
                batch_label_masks.append(label_mask)

            batch_text_ids = np.asarray(batch_text_ids, dtype=int)
            batch_labels = np.asarray(batch_labels, dtype=int)
            batch_mask = np.asarray(batch_mask, dtype=int)
            batch_label_masks = np.asarray(batch_label_masks, dtype=int)

            batches_texts.append(batch_texts)
            batches_text_ids.append(batch_text_ids)
            batches_labels.append(batch_labels)
            batches_mask.append(batch_mask)
            batches_label_masks.append(batch_label_masks)

            i += batch_size

        return batches_texts, batches_text_ids, batches_labels,\
            batches_mask, batches_label_masks
