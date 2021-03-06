import torch
from torch.utils.data import Dataset

from constants import DEVICE
from criterion.loss import LossWithLS
from model.transformers.transformers import Transformer
from optimizer.optimizer import AdamWarmup
from ultis import create_masks


class Trainer:
    def __init__(
            self,
            d_model: int = 512,
            heads: int = 4,
            num_layers: int = 3,
            epochs: int = 10,
            batch_size: int = 4,
            word_map=None,
            dataset: Dataset = None
    ):
        super(Trainer, self).__init__()

        if word_map is None:
            word_map = {}
        self.train_loader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True)

        self.d_model = d_model
        self.heads = heads
        self.num_layers = num_layers
        self.device = DEVICE
        self.epochs = epochs
        self.word_map = word_map

        self.transformer = Transformer(d_model=self.d_model, heads=self.heads, num_layers=self.num_layers,
                                       word_map=self.word_map)
        self.transformer = self.transformer.to(self.device)
        self.adam_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.transformer_optimizer = AdamWarmup(model_size=self.d_model, warmup_steps=4000,
                                                optimizer=self.adam_optimizer)
        self.criterion = LossWithLS(len(self.word_map), 0.1)

    def train(self, epoch):
        self.transformer.train()
        sum_loss = 0
        count = 0

        for i, (question, reply) in enumerate(self.train_loader):

            samples = question.shape[0]

            # Move to device
            question = question.to(self.device)
            reply = reply.to(self.device)

            # Prepare Target Data
            reply_input = reply[:, :-1]
            reply_target = reply[:, 1:]

            # Create mask and add dimensions
            question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

            # Get the transformer outputs
            out = self.transformer(question, question_mask, reply_input, reply_input_mask)

            # Compute the loss
            loss = self.criterion(out, reply_target, reply_target_mask)

            # Backprop
            self.transformer_optimizer.optimizer.zero_grad()
            loss.backward()
            self.transformer_optimizer.step()

            sum_loss += loss.item() * samples
            count += samples

            if i % 100 == 0:
                print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(self.train_loader), sum_loss / count))

        for epoch in range(self.epochs):
            self.train(epoch)

            state = {'epoch': epoch, 'transformer': self.transformer,
                     'transformer_optimizer': self.transformer_optimizer}
            torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')
