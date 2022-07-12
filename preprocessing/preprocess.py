import json
from collections import Counter

from constants import CORPUS_MOVIE_CONV, MAX_LEN, CORPUS_MOVIE_LINES


def remove_punc(string):
    punctuations = '''!()-[]{};:'",<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char  # space is also a character
    return no_punct.lower()


def preprocessing():
    with open(CORPUS_MOVIE_CONV, 'r') as c:
        conv = c.readlines()

    with open(CORPUS_MOVIE_LINES, 'r') as f:
        lines = f.readlines()

    lines_dic = {}
    for line in lines:
        objects = line.split(" +++$+++ ")
        lines_dic[objects[0]] = objects[-1]

    pairs = []
    for con in conv:
        ids = eval(con.split(" +++$+++ ")[-1])
        for i in range(len(ids)):
            qa_pairs = []

            if i == len(ids) - 1:
                break

            first = remove_punc(lines_dic[ids[i]].strip())
            second = remove_punc(lines_dic[ids[i + 1]].strip())
            qa_pairs.append(first.split()[:MAX_LEN])
            qa_pairs.append(second.split()[:MAX_LEN])
            pairs.append(qa_pairs)

    word_freq = Counter()
    for pair in pairs:
        word_freq.update(pair[0])
        word_freq.update(pair[1])

    min_word_freq = 5
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    print("Total words are: {}".format(len(word_map)))

    with open('WORDMAP_corpus.json', 'w') as j:
        json.dump(word_map, j)

    pairs_encoded = []
    for pair in pairs:
        qus = encode_question(pair[0], word_map)
        ans = encode_reply(pair[1], word_map)
        pairs_encoded.append([qus, ans])

    with open('pairs_encoded.json', 'w') as p:
        json.dump(pairs_encoded, p)


def encode_question(words, word_map):
    enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (MAX_LEN - len(words))
    return enc_c


def encode_reply(words, word_map):
    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<end>']] + [
        word_map['<pad>']] * (MAX_LEN - len(words))
    return enc_c
