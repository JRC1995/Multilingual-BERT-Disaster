import pickle
import numpy as np
from fasttext import load_model

with open("../Processed_Data/vocab2count.pkl", "rb") as fp:
    vocab2count = pickle.load(fp)

ft_model = load_model("../Embeddings/crawl-300d-2M-subword.bin")

word_vec_dim = 200


def loadEmbeddings(filename):
    vocab2embd = {}

    global word_vec_dim

    with open(filename) as infile:
        for line in infile:
            row = line.strip().split(' ')
            word = row[0].lower()
            # print(word)
            if word not in vocab2embd:
                vec = np.asarray(row[1:], np.float32)
                if len(vec) == word_vec_dim:
                    vocab2embd[word] = vec

    print('Embedding Loaded.')
    return vocab2embd


def loadEmbeddings(filename):
    vocab2embd = {}

    global word_vec_dim

    with open(filename) as infile:
        for line in infile:
            row = line.strip().split(' ')
            word = row[0]
            # print(word)
            if word not in vocab2embd:
                vec = np.asarray(row[1:], np.float32)
                if len(vec) == word_vec_dim:
                    vocab2embd[word] = vec

    print('Embedding Loaded.')
    return vocab2embd


vocab2embd = loadEmbeddings("../Embeddings/glove.twitter.27B.200d.txt")

special_tags = ["<UNK>", "<USER>", "<PAD>", "<URL>", "<HASH>", "</HASH>"]

vocab2count = {v: c for v, c in vocab2count.items() if v not in special_tags}

vocab = []
counter = []

for word, count in vocab2count.items():
    counter.append(count)
    vocab.append(word)

vocab_idx = np.argsort(counter).tolist()
vocab_idx.reverse()

vocab = [vocab[id] for id in vocab_idx]

if len(vocab) > 40000-len(special_tags):
    vocab = vocab[0:40000-len(special_tags)]

vocab += special_tags

# print(vocab[0:100])
# print(vocab)

np.random.seed(101)

USER = np.random.randn(word_vec_dim)
UNK = np.random.randn(word_vec_dim)
URL = np.random.randn(word_vec_dim)
HASH = np.random.randn(word_vec_dim)
_HASH = np.random.randn(word_vec_dim)
PAD = np.zeros((word_vec_dim,), np.float32)


def embed(word):

    global USER
    global UNK
    global PAD
    global URL
    global HASH
    global _HASH
    global word_vec_dim
    global special_tags
    global vocab2embd

    if word in special_tags or word not in vocab2embd:
        if word == "<USER>":
            return USER
        elif word == "<PAD>":
            return PAD
        elif word == "<UNK>":
            return UNK
        elif word == "<URL>":
            return URL
        elif word == "<HASH>":
            return HASH
        elif word == "</HASH>":
            return _HASH
        else:
            return np.random.randn(word_vec_dim)
    else:
        return vocab2embd[word]


def ft_embed(word):

    global ft_model

    if word == "<PAD>":
        return np.zeros((300), np.float32)
    else:
        return ft_model.get_word_vector(word)


vocab2idx = {v: i for i, v in enumerate(vocab)}
# print(vocab2idx)
embeddings = np.asarray([embed(v.lower()) for v in vocab], np.float32)
ft_embeddings = np.asarray([ft_embed(v) for v in vocab], np.float32)

d = {}
d['vocab2idx'] = vocab2idx
d['embeddings'] = embeddings
d['ft_embeddings'] = ft_embeddings

with open("../Processed_Data/vocab.pkl", "wb") as fp:
    pickle.dump(d, fp)
