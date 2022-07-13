import json

from criterion.loss import LossWithLS
from dataset.datasets import DataSet
from model.transformers.transformers import Transformer
from preprocessing.preprocess import preprocessing
from trainer.trainer import Trainer


def train():
    preprocessing()
    trainer = Trainer()

    dataloader = DataSet()
    with open('data/WORDMAP_corpus.json', 'r') as j:
        word_map = json.load(j)
    loss = LossWithLS(word_map, 0.1)
    transformer = Transformer(word_map=word_map)
    trainer.train(criterion=loss, train_loader=dataloader, transformer=transformer, epoch=10)


if __name__ == '__main__':
    train()
