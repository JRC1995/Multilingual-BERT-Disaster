
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from transformers import *
import random


class Classifier(nn.Module):
    def __init__(self, classes_num, config, device):
        super(Classifier, self).__init__()
        self.BERT = BertModel.from_pretrained('../Embeddings/Pre_trained_BERT/',
                                              output_hidden_states=True,
                                              output_attentions=False)

        BERT_out_dropout = config["output_dropout"]

        self.dropout = nn.Dropout(BERT_out_dropout)

        self.linear = nn.Linear(768, classes_num)

    # @torchsnooper.snoop()
    def forward(self, x, mask, lam=None, shuffled_indices=None):

        N, S = x.size()

        _, pooled_output, _ = self.BERT(x, attention_mask=mask)

        pooled_output = self.dropout(pooled_output)

        logits = T.sigmoid(self.linear(pooled_output))

        return logits
