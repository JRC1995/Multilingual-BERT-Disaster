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
        self.ks = 3  # There are three conv nets here
        self.pad_idx = pad_idx

        input_channel = 1

        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (3, words_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (4, words_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (5, words_dim), padding=(4, 0))

        self.dropout = nn.Dropout(config['dropout'])
        self.fc1 = nn.Linear(self.ks*self.output_channel, classes_num)

    def forward(self, x, mask):

        embeddings_dropout_mask = self.embedding_dropout(self.embedding_ones)
        dropped_embeddings = self.embeddings*embeddings_dropout_mask

        x = F.embedding(x, dropped_embeddings, padding_idx=self.pad_idx)

        x = x.unsqueeze(1)

        x = [F.relu(self.conv1(x)).squeeze(3),
             F.relu(self.conv2(x)).squeeze(3),
             F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling

        # (batch, channel_output*dynamic_pool_length) * ks
        x = T.cat(x, 1)  # (batch, channel_output * ks)
        x = self.dropout(x)
        logit = T.sigmoid(self.fc1(x))  # (batch, target_size)
        return logit
