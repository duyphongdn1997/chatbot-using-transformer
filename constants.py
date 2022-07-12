import torch

CORPUS_MOVIE_CONV = 'data/movie_conversations.txt'
CORPUS_MOVIE_LINES = 'data/movie_lines.txt'
MAX_LEN = 25
LOAD_CHECKPOINT = True
CKPT_PATH = 'models/checkpoint.pth.tar'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
