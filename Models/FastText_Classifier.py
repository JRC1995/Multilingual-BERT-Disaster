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

        self.linear = nn.Linear(dim, config["hidden_size"])
        self.out_linear = nn.Linear(config["hidden_size"], classes_num)

    def forward(self, input, mask):

        embeddings_dropout_mask = self.embedding_dropout(self.embedding_ones)

        dropped_embeddings = self.embeddings*embeddings_dropout_mask

        embedded_input = F.embedding(input, dropped_embeddings, padding_idx=self.pad_idx)

        N, S, D = embedded_input.size()
        mask = mask.view(N, S, 1)

        input_dropout_mask = self.input_dropout(mask)

        embedded_input = embedded_input*input_dropout_mask

        if S < self.n_gram:
            avg_pooled_out = embedded_input
        else:
            grammed_input = []
            for i in range(S-self.n_gram+1):
                n_gram_input = embedded_input[:, i:i+self.n_gram, :]
                grammed_input.append(n_gram_input.view(N, 1, self.n_gram, D))
            grammed_input = T.cat(grammed_input, dim=1)
            avg_pooled_out = T.mean(grammed_input, dim=-2)

        max_pooled_out, _ = T.max(avg_pooled_out, dim=1)
        max_pooled_out = self.output_dropout(max_pooled_out)
        intermediate = F.relu(self.linear(max_pooled_out))
        logits = T.sigmoid(self.out_linear(intermediate))

        return logits
