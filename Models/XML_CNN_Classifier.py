# Adapted from: https://github.com/castorini/hedwig/blob/master/models/xml_cnn/model.py

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, embeddings, pad_idx, classes_num,
                 config, device):
        super(Classifier, self).__init__()

        self.embeddings = nn.Parameter(T.tensor(embeddings).float().to(device))
        self.embedding_dropout = nn.Dropout(config['embedding_dropout'])
        self.embedding_ones = T.ones(self.embeddings.size(0), 1).float().to(device)

        self.output_channel = config['output_channel']
        words_dim = self.embeddings.size(-1)
        self.num_bottleneck_hidden = config['hidden_size']
        self.dynamic_pool_length = config['dynamic_pool_length']
        self.ks = 3  # There are three conv nets here
        self.pad_idx = pad_idx

        input_channel = 1

        # Different filter sizes in xml_cnn than kim_cnn
        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (2, words_dim), padding=(1, 0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (4, words_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (8, words_dim), padding=(7, 0))

        self.dropout = nn.Dropout(config['dropout'])
        self.bottleneck = nn.Linear(self.ks * self.output_channel *
                                    self.dynamic_pool_length, self.num_bottleneck_hidden)
        self.fc1 = nn.Linear(self.num_bottleneck_hidden, classes_num)

        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)  # Adaptive pooling

    def forward(self, x, mask):

        embeddings_dropout_mask = self.embedding_dropout(self.embedding_ones)
        dropped_embeddings = self.embeddings*embeddings_dropout_mask

        x = F.embedding(x, dropped_embeddings, padding_idx=self.pad_idx)

        x = x.unsqueeze(1)

        x = [F.relu(self.conv1(x)).squeeze(3),
             F.relu(self.conv2(x)).squeeze(3),
             F.relu(self.conv3(x)).squeeze(3)]

        x = [self.pool(i).squeeze(2) for i in x]

        # (batch, channel_output*dynamic_pool_length) * ks
        x = T.cat(x, 1)  # (batch, channel_output * ks)
        x = F.relu(self.bottleneck(x.view(-1, self.ks * self.output_channel * self.dynamic_pool_length)))
        x = self.dropout(x)
        logit = T.sigmoid(self.fc1(x))  # (batch, target_size)
        return logit
