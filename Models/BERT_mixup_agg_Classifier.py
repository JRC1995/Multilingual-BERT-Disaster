
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from Lib.Encoders.BERT_mixup import *
import random


class Classifier(nn.Module):
    def __init__(self, classes_num, config, device):
        super(Classifier, self).__init__()
        self.BERT = BertModel.from_pretrained('../Embeddings/Pre_trained_BERT/',
                                              output_hidden_states=True,
                                              output_attentions=False)

        BERT_out_dropout = config["output_dropout"]

        self.n_fold = config['n_fold']

        self.manifold_mixup = config["manifold_mixup"]
        self.dropout = nn.Dropout(BERT_out_dropout)

        self.linear = nn.Linear(self.n_fold*768, self.n_fold*classes_num)

    def mixup(x, current_layer, mixup_dict, training):

        if training:
            layer_num = mixup_dict["layer_num"]
            if current_layer == layer_num:
                shuffled_indices = mixup_dict["shuffled_indices"]
                lam = mixup_dict["lam"]
                x_shuffled = x[shuffled_indices]
                mixed_x = lam*x + (1-lam)*x_shuffled
                return mixed_x
            else:
                return x
        else:
            return x

    # @torchsnooper.snoop()
    def forward(self, x, mask, mixup_dict=None, shuffled_indices=None):

        N, S = x.size()

        _, pooled_output, _ = self.BERT(x, attention_mask=mask, mixup_dict=mixup_dict)

        pooled_output = mixup(pooled_output, 12, mixup_dict, self.training)

        pooled_output = self.dropout(pooled_output)

        pooled_outputs = [pooled_output]

        for i in range(self.n_fold - 1):
            if shuffled_indices is None:
                pooled_outputs.append(pooled_output)
            else:
                pooled_outputs.append(pooled_output[shuffled_indices[i]])

        agg_pooled_output = T.cat(pooled_outputs, dim=-1)

        logits = T.sigmoid(self.linear(agg_pooled_output))

        return logits
