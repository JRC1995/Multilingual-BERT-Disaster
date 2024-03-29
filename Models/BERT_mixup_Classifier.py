
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

        #self.manifold_mixup = config["manifold_mixup"]
        self.dropout = nn.Dropout(BERT_out_dropout)

        self.linear = nn.Linear(768, classes_num)

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
    def forward(self, x, mask, mixup_dict=None):

        N, S = x.size()

        _, pooled_output, _ = self.BERT(x, attention_mask=mask, mixup_dict=mixup_dict)

        pooled_output = mixup(pooled_output, 12, mixup_dict, self.training)

        pooled_output = self.dropout(pooled_output)

        logits = T.sigmoid(self.linear(pooled_output))

        return logits
