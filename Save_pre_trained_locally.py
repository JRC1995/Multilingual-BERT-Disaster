import torch
from transformers import *

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,      BertTokenizer,        'bert-base-multilingual-cased')]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:
for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True,
                                        output_attentions=False)

    print(len(tokenizer))

    special_tokens_dict = {'additional_special_tokens': ['<USER>', '<URL>', '<HASH>', '</HASH>']}

    tokenizer.add_special_tokens(special_tokens_dict)

    print(len(tokenizer))

    model.resize_token_embeddings(len(tokenizer))

    model.save_pretrained('Embeddings/Pre_trained_BERT/')  # save
    tokenizer.save_pretrained('Embeddings/Pre_trained_BERT/')  # save
