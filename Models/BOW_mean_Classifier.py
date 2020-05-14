import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, embeddings, pad_idx, classes_num,
                 config, device):

        super(Classifier, self).__init__()

        self.pad_idx = pad_idx
        self.n_gram = config["n_gram"]

        self.embeddings = nn.Parameter(T.tensor(embeddings).float().to(device))
        dim = self.embeddings.size()[1]
        self.PAD = T.zeros(1, self.n_gram//2, dim).float().to(device)
        self.embedding_ones = T.ones(self.embeddings.size(0), 1).float().to(device)

        self.embedding_dropout = nn.Dropout(config["embedding_dropout"])
        self.input_dropout = nn.Dropout(config["input_dropout"])
        self.output_dropout = nn.Dropout(config["output_dropout"])

        self.out_linear = nn.Linear(dim, classes_num)

    def forward(self, input, mask):

        embeddings_dropout_mask = self.embedding_dropout(self.embedding_ones)

        dropped_embeddings = self.embeddings*embeddings_dropout_mask

        embedded_input = F.embedding(input, dropped_embeddings, padding_idx=self.pad_idx)

        N, S, D = embedded_input.size()
        mask = mask.view(N, S, 1)

        input_dropout_mask = self.input_dropout(mask)

        embedded_input = embedded_input*input_dropout_mask

        avg_input = embedded_input.mean(dim=1)

        logits = T.sigmoid(self.out_linear(avg_input))

        return logits
