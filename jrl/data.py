import gzip
import math
import os
import os.path as osp
import random
from typing import Dict

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class MyWeightedSampler(Sampler):

    def __init__(self, weights, num_samples=None, shuffle=False):
        self.weights = weights
        self.num_samples = num_samples if num_samples else weights.shape[0]

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for i in range(self.weights.shape[0]):
            if random.random() < self.weights[i]:
                yield i


class ConcatDataset(Dataset):

    def __init__(self, datasets: list):
        self.datasets = datasets

    def __len__(self):
        return max([len(data) for data in self.datasets])

    def __getitem__(self, item):
        ret = []
        for data in self.datasets:
            mode = data.mode
            data.mode = 'train'
            ret.extend(data[item % len(data)])
            data.mode = mode
        return ret


class JRLDataLoader(Dataset):

    def __init__(self, cfg: Dict, mode):
        super().__init__()
        self.cfg = cfg
        self.data_path = data_path = cfg.DATA.PATH
        self.need_image = cfg.MODEL.NEED_IMAGE
        self.mode = mode  #cfg.DATA.MODE

        self.data = np.load(osp.join(data_path, f'{mode}_data.npy'),
                            mmap_mode="r")
        self.data_weights = np.load(osp.join(data_path,
                                             f'{mode}_data_weights.npy'),
                                    mmap_mode="r")
        attrs = [
            'vocab_size', 'user_size', 'user_ids', 'product_size',
            'product_ids', 'review_size', 'vocab_distribute',
            'review_distribute', 'product_distribute'
        ]
        for att in attrs:
            setattr(self, att, np.load(osp.join(data_path,
                                                f"{mode}_{att}.npy")))

        if mode == 'test':
            self.user_train_product_set_list = self._read_train_product_ids(
                self.user_size)

        print("Data statistic: vocab %d, review %d, user %d, product %d\n" %
              (self.vocab_size, self.review_size, self.user_size,
               self.product_size))

    def _read_train_product_ids(self, user_size):
        user_train_product_set_list = [set() for i in range(user_size)]
        train_review_size = 0
        with gzip.open(self.cfg.DATA.PATH + 'train.txt.gz', 'rt') as fin:
            for line in fin:
                train_review_size += 1
                arr = line.strip().split('\t')
                user_train_product_set_list[int(arr[0])].add(int(arr[1]))
        return user_train_product_set_list

    def __len__(self):
        if self.mode == 'train':
            return self.data.shape[0]
        elif self.mode == 'test':
            return self.user_size

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = torch.tensor(self.data[idx], dtype=torch.long)

            if self.need_image:
                img_features = torch.load(
                    os.path.join(self.data_path,
                                 f'img_features_{data[1]}.pth.tar'))
                return data, img_features

            return data, torch.tensor(0)
        else:
            return idx

    def set_mode(self, mode):
        assert mode in ['train', 'test']
        self.mode = mode

    def get_sizes(self):
        sizes = edict()
        sizes.vocab_size = self.vocab_size
        sizes.user = self.user_size
        sizes.product = self.product_size
        sizes.review = self.review_size
        return sizes

    def get_distributions(self):
        distributions = edict()
        distributions.words = self.vocab_distribute
        distributions.reviews = self.review_distribute
        distributions.products = self.product_distribute
        return distributions

    def del_distributions(self):
        distributions = edict()
        del self.vocab_distribute
        del self.review_distribute
        del self.product_distribute

    def get_features(self):
        if self.cfg.MODEL.NEED_IMAGE:
            features = edict()
            features.image = self.img_features
            return features
        return None

    def compute_test_product_ranklist(self, u_idx, original_scores,
                                      sorted_product_idxs, rank_cutoff):
        product_rank_list = []
        product_rank_scores = []
        rank = 0
        for product_idx in sorted_product_idxs:
            if product_idx in self.user_train_product_set_list[
                    u_idx] or math.isnan(original_scores[product_idx]):
                continue
            product_rank_list.append(product_idx)
            product_rank_scores.append(original_scores[product_idx])
            rank += 1
            if rank == rank_cutoff:
                break
        return product_rank_list, product_rank_scores

    def output_ranklist(self, user_ranklist_map, user_ranklist_score_map,
                        output_path, similarity_func):
        with open(
                os.path.join(output_path,
                             'test.' + similarity_func + '.ranklist'),
                'w') as rank_fout:
            for u_idx in user_ranklist_map:
                user_id = self.user_ids[u_idx]
                for i in range(len(user_ranklist_map[u_idx])):
                    product_id = self.product_ids[user_ranklist_map[u_idx][i]]
                    rank_fout.write(user_id + ' Q0 ' + product_id + ' ' +
                                    str(i + 1) + ' ' +
                                    str(user_ranklist_score_map[u_idx][i]) +
                                    ' MultiViewEmbedding\n')
