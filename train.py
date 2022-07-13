from dataset.datasets import DataSet
from preprocessing.preprocess import preprocessing
from trainer.trainer import Trainer


def train():
    preprocessing()

    dataloader = DataSet()
    trainer = Trainer(dataset=dataloader)
    trainer.train(epoch=10)


if __name__ == '__main__':
    train()
