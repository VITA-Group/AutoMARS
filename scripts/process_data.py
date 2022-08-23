import gzip
import os.path as osp
import sys

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm


def generate_data(data_path, set_name, sub_sampling_rate):
    variables = edict()  #collections.defaultdict()
    #get product/user/vocabulary information
    variables.product_ids = []
    with gzip.open(data_path + 'product.txt.gz', 'rt') as fin:
        for line in tqdm(fin, desc='product'):
            variables.product_ids.append(line.strip())
    variables.product_size = len(variables.product_ids)
    variables.user_ids = []
    with gzip.open(data_path + 'users.txt.gz', 'rt') as fin:
        for line in tqdm(fin, desc='users'):
            variables.user_ids.append(line.strip())
    variables.user_size = len(variables.user_ids)
    variables.words = []
    with gzip.open(data_path + 'vocab.txt.gz', 'rt') as fin:
        for line in tqdm(fin, desc='vocabs'):
            variables.words.append(line.strip())
    variables.vocab_size = len(variables.words)

    #get review sets
    variables.word_count = 0
    variables.vocab_freq = np.zeros(variables.vocab_size)
    variables.review_info = []
    variables.review_text = []
    variables.review_words_size = 0
    with gzip.open(data_path + set_name + '.txt.gz', 'rt') as fin:
        for line in tqdm(fin, desc='reviews'):
            arr = line.strip().split('\t')
            variables.review_info.append(
                (int(arr[0]), int(arr[1])))  # (user_idx, product_idx)
            variables.review_text.append([int(i) for i in arr[2].split(' ')])
            variables.review_words_size += len(variables.review_text[-1])
            for idx in variables.review_text[-1]:
                variables.vocab_freq[idx] += 1
            variables.word_count += len(variables.review_text[-1])

    variables.review_size = len(variables.review_info)
    variables.vocab_distribute = variables.vocab_freq / sum(
        variables.vocab_freq)
    variables.sub_sampling_rate = sub_sampling(sub_sampling_rate,
                                               variables.vocab_size,
                                               variables.vocab_freq)
    variables.review_distribute = np.ones(
        variables.review_size) / variables.review_size
    variables.product_distribute = np.ones(
        variables.product_size) / variables.product_size
    data = []  #np.array([], dtype=np.uint32)
    data_probability = []
    for i, review in tqdm(enumerate(variables.review_text),
                          desc='data',
                          total=len(variables.review_text)):
        user_idx = variables.review_info[i][0]
        product_idx = variables.review_info[i][1]
        review_idx = i
        for word_idx in review:
            data.append([user_idx, product_idx, review_idx, word_idx])
            data_probability.append(variables.sub_sampling_rate[word_idx])
    variables.data = np.array(data, dtype=np.int32)
    variables.data_weights = np.array(data_probability)

    attributes = [
        'data', 'data_weights', 'vocab_distribute', 'review_distribute',
        'product_distribute', 'user_size', 'user_ids', 'vocab_size',
        'review_size', 'product_size', 'product_ids'
    ]
    keys = list(variables.keys())
    for k in keys:
        if k in attributes:
            np.save(osp.join(data_path, f"{set_name}_{k}.npy"), variables[k])


def sub_sampling(subsample_threshold, vocab_size, vocab_distribute):
    sub_sampling_rate = np.ones(vocab_size)
    if subsample_threshold == 0.0:
        return None
    threshold = sum(vocab_distribute) * subsample_threshold
    count_sub_sample = 0
    for i in range(vocab_size):
        if vocab_distribute[i] == 0.0:
            sub_sampling_rate[i] = 0.0
            continue
        sub_sampling_rate[i] = np.sqrt(
            float(vocab_distribute[i]) / threshold) + 1
        sub_sampling_rate[i] = min(
            sub_sampling_rate[i] * threshold / vocab_distribute[i], 1)
        count_sub_sample += 1
    return sub_sampling_rate


if __name__ == "__main__":

    _, path = sys.argv
    generate_data(path, 'train', 0.0001)
    generate_data(path, 'test', 0.0001)
