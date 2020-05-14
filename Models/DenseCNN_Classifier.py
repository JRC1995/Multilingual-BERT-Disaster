# Adapted from: https://github.com/castorini/hedwig/tree/master/models/kim_cnn

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, embeddings, pad_idx, classes_num,
                 config, device):
        super(Classifier, self).__init__()

        trainable_embeddings = config['trainable_embeddings']

        if trainable_embeddings:
            self.embeddings = nn.Parameter(T.tensor(embeddings).float().to(device))
        else:
            self.embeddings = T.tensor(embeddings).float().to(device)

        self.embedding_dropout = nn.Dropout(config['embedding_dropout'])
        self.embedding_ones = T.ones(self.embeddings.size(0), 1).float().to(device)

        self.output_channel = config['output_channel']
        words_dim = self.embeddings.size(-1)
        self.pad_idx = pad_idx

        self.fc0 = nn.Linear(words_dim, self.output_channel)

        self.conv1 = nn.Conv1d(self.output_channel, self.output_channel, 3,
                               padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm1d(self.output_channel)
        self.conv2 = nn.Conv1d(2*self.output_channel, self.output_channel, 3,
                               padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm1d(self.output_channel)
        self.conv3 = nn.Conv1d(3*self.output_channel, self.output_channel, 3,
                               padding=1, padding_mode='zeros')
        self.bn3 = nn.BatchNorm1d(self.output_channel)

        """
        self.conv4 = nn.Conv1d(4*self.output_channel, self.output_channel, 3,
                               padding=1, padding_mode='zeros')
        self.bn4 = nn.BatchNorm1d(self.output_channel)
        self.conv5 = nn.Conv1d(5*self.output_channel, self.output_channel, 3,
                               padding=1, padding_mode='zeros')
        self.bn5 = nn.BatchNorm1d(self.output_channel)
        self.conv6 = nn.Conv1d(6*self.output_channel, self.output_channel, 3,
                               padding=1, padding_mode='zeros')
        self.bn6 = nn.BatchNorm1d(self.output_channel)
        """

        self.attn_linear1 = nn.Linear(1, 10)
        self.attn_linear2 = nn.Linear(10, 1)

        self.dropout = nn.Dropout(config['dropout'])
        self.fc1 = nn.Linear(self.output_channel, config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], classes_num)

    def forward(self, x, mask):

        embeddings_dropout_mask = self.embedding_dropout(self.embedding_ones)
        dropped_embeddings = self.embeddings*embeddings_dropout_mask

        layers = []

        x = F.embedding(x, dropped_embeddings, padding_idx=self.pad_idx)

        x = F.relu(self.fc0(x))
        x = T.transpose(x, 1, 2)  # N x C x S
        layers.append(x)

        x_ = F.relu(self.bn1(self.conv1(x)))
        layers.append(x_)
        x = T.cat([x, x_], dim=1)

        x_ = F.relu(self.bn2(self.conv2(x)))
        layers.append(x_)
        x = T.cat([x, x_], dim=1)

        x_ = F.relu(self.bn3(self.conv3(x)))
        layers.append(x_)
        """
        x = T.cat([x, x_], dim=1)

        x_ = F.relu(self.bn4(self.conv4(x)))
        layers.append(x_)
        x = T.cat([x, x_], dim=1)

        x_ = F.relu(self.bn5(self.conv5(x)))
        layers.append(x_)
        x = T.cat([x, x_], dim=1)

        x_ = F.relu(self.bn6(self.conv6(x)))
        layers.append(x_)
        """

        s_l = [x_.sum(dim=1).unsqueeze(1) for x_ in layers]
        s_l = T.cat(s_l, dim=1)
        s_l = T.transpose(s_l, 1, 2).unsqueeze(-1)

        a = self.attn_linear2(T.tanh(self.attn_linear1(s_l))).squeeze(-1)
        a = F.softmax(T.transpose(a, 1, 2), dim=1).unsqueeze(2)
        x_l = T.cat([x_.unsqueeze(1) for x_ in layers], dim=1)
        attended = T.sum(a*x_l, dim=1)

        x, _ = T.max(attended, dim=-1)

        x = self.dropout(x)

        logit = T.sigmoid(self.fc2(F.relu(self.fc1(x))))  # (batch, target_size)
        return logit
