
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
        self.n_grams = config["n_grams"]

        self.manifold_mixup = config["manifold_mixup"]
        self.dropout = nn.Dropout(BERT_out_dropout)

        self.linear = nn.Linear(768, config["hidden_size"])
        self.out_linear = nn.Linear(config["hidden_size"], classes_num)

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

    def hierarchical_pooling(x, n_grams=3):
        S = x.size(1)
        if S < n_grams:
            avg_pooled_out = x
        else:
            grammed_input = []
            for i in range(S-n_grams+1):
                n_gram_input = x[:, i:i+n_grams, :]
                grammed_input.append(n_gram_input.view(N, 1, n_grams, D))
            grammed_input = T.cat(grammed_input, dim=1)
            avg_pooled_out = T.mean(grammed_input, dim=-2)

        max_pooled_out, _ = T.max(avg_pooled_out, dim=1)

        return max_pooled_out

    # @torchsnooper.snoop()

    def forward(self, x, mask, mixup_dict=None):

        N, S = x.size()

        _, _, all_hidden_states = self.BERT(x, attention_mask=mask)

        pooled_output = hierarchical_pooling(all_hidden_states, self.n_grams)
        pooled_output = self.dropout(pooled_output)
        intermediate = F.relu(self.linear(pooled_output))

        intermediate = mixup(intermediate, 12, mixup_dict, self.training)

        logits = T.sigmoid(self.out_linear(intermediate))

        return logits
