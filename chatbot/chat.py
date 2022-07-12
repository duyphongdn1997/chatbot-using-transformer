import json

import torch.utils.data

from ultis import *
from constants import DEVICE, CKPT_PATH
from evaluate.evalute import evaluate


def main(load_checkpoint, word_map):
    if load_checkpoint:
        checkpoint = torch.load(CKPT_PATH)
        transformer = checkpoint['transformer']

    while True:
        question = input("Question: ")
        if question == 'quit':
            break
        max_len = input("Maximum Reply Length: ")
        enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
        question = torch.LongTensor(enc_qus).to(DEVICE).unsqueeze(0)
        question_mask = (question != 0).to(DEVICE).unsqueeze(1).unsqueeze(1)
        sentence = evaluate(transformer, question, question_mask, int(max_len), word_map)
        print(sentence)


if __name__ == "__main__":
    with open('WORDMAP_corpus.json', 'r') as j:
        word_map = json.load(j)
    main(load_checkpoint=True, word_map=word_map)
