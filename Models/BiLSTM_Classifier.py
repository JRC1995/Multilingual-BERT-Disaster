import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from Lib.Encoders.BiLSTM import BiLSTM
import json


class Classifier(nn.Module):
    def __init__(self, embeddings, pad_idx, classes_num,
                 config, device):

        super(Classifier, self).__init__()

        trainable_embeddings = config["trainable_embeddings"]

        if trainable_embeddings:
            self.embeddings = nn.Parameter(T.tensor(embeddings).float().to(device))
        else:
            self.embeddings = T.tensor(embeddings).float().to(device)

        self.pad_idx = pad_idx

        input_dim = self.embeddings.size()[1]
        dim = config["input_dim"]
        hidden_size = config["hidden_size"]
        layers = config["layers"]
        embedding_dropout = config["embedding_dropout"]
        input_dropconnect = config["input_dropconnect"]
        hidden_dropconnect = config["hidden_dropconnect"]
        zoneout = config["zoneout"]
        output_dropout = config["output_dropout"]
        parameterized_states = config["parameterized_states"]
        input_dropout = config["input_dropout"]

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.embedding_ones = T.ones(self.embeddings.size(0), 1).float().to(device)

        self.transform_embeddings = config["transform_embeddings"]

        if self.transform_embeddings:
            self.transform_linear = nn.Linear(input_dim, dim)

        self.input_dropout = nn.Dropout(input_dropout)

        self.BiLSTM_layers = []

        self.BiLSTM_layers.append(BiLSTM(D=dim, hidden_size=hidden_size,
                                         input_dropconnect=input_dropconnect,
                                         hidden_dropconnect=hidden_dropconnect,
                                         zoneout=zoneout,
                                         parameterized_states=parameterized_states,
                                         device=device))

        if layers > 1:
            for layer in range(layers-1):
                self.BiLSTM_layers.append(BiLSTM(D=2*hidden_size, hidden_size=hidden_size,
                                                 input_dropconnect=input_dropconnect,
                                                 hidden_dropconnect=hidden_dropconnect,
                                                 zoneout=zoneout,
                                                 parameterized_states=parameterized_states,
                                                 device=device))

        for layer in range(layers):
            for name, param in self.BiLSTM_layers[layer].named_parameters():
                self.register_parameter("BiLSTM_"+str(layer)+"_"+name, param)

        self.linear = nn.Linear(2*hidden_size, classes_num)

        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, input, mask):

        embeddings_dropout_mask = self.embedding_dropout(self.embedding_ones)
        dropped_embeddings = self.embeddings*embeddings_dropout_mask

        embedded_input = F.embedding(input, dropped_embeddings, padding_idx=self.pad_idx)

        if self.transform_embeddings:
            LSTM_in = self.transform_linear(embedded_input)
        else:
            LSTM_in = embedded_input

        LSTM_in = self.input_dropout(LSTM_in)

        for BiLSTM in self.BiLSTM_layers:
            LSTM_in, _ = BiLSTM(LSTM_in, mask)

        hidden_states = LSTM_in

        max_pooled_out, _ = T.max(hidden_states, dim=1)

        max_pooled_out = self.output_dropout(max_pooled_out)

        logits = T.sigmoid(self.linear(max_pooled_out))

        return logits


"""
with open("../Configs/BiLSTM_config.json", "r") as file:
    config = json.load(file)

embeddings = np.random.randn(1000, 200)


model = Classifier(embeddings, 0, 10, config, 'cuda')

for name, param in model.named_parameters():
    print(name)
"""
