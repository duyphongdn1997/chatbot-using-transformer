import math

import torch
import torch.nn as nn

from constants import DEVICE


class Embeddings(nn.Module):
    """
    Implements embeddings of the words and adds their positional encodings.
    """

    def __init__(self, vocab_size, d_model, max_len=50):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positinal_encoding(max_len, self.d_model)
        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def create_positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(DEVICE)
        for pos in range(max_len):  # for each position of the word
            for i in range(0, d_model, 2):  # for each dimension of the a position
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)  # include the batch size
        return pe

    def forward(self, encoded_words):
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)
        embedding += self.pe[:,
                     :embedding.size(1)]  # pe will automatically be expanded with the same batch size as encoded_words
        embedding = self.dropout(embedding)
        return embedding
