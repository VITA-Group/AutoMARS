import os
import time
from collections import defaultdict as ddict

import pandas as pd
import torch


def print_metrics_with_rank_cutoff(rank_cutoff, qrel_map, rank_list_file):
    #read rank_list file
    rank_list = {}
    with open(rank_list_file) as fin:
        for line in fin:
            arr = line.strip().split(' ')
            qid = arr[0]
            did = arr[2]
            if qid not in rank_list:
                rank_list[qid] = []
            if len(rank_list[qid]) > rank_cutoff:
                continue
            rank_list[qid].append(did)

    ndcgs = 0.0
    recalls = 0.0
    hits = 0.0
    large_rels = 0.0
    precisions = 0.0
    count_query = 0
    for qid in rank_list:
        if qid in qrel_map:
            recall, ndcg, hit, large_rel, precision = metrics(
                rank_list[qid], qrel_map[qid])
            count_query += 1
            ndcgs += ndcg
            recalls += recall
            hits += hit
            large_rels += large_rel
            precisions += precision

    print("Query Number:" + str(count_query))
    print("Larger_rel_set@" + str(rank_cutoff) + ":" +
          str(large_rels / count_query))
    print("Recall@" + str(rank_cutoff) + ":" + str(recalls / count_query))
    print("Precision@" + str(rank_cutoff) + ":" +
          str(precisions / count_query))
    print("NDCG@" + str(rank_cutoff) + ":" + str(ndcgs / count_query))
    print("Hit@" + str(rank_cutoff) + ":" + str(hits / count_query))

    results = {
        "ndcgs": ndcgs / count_query,
        'recalls': recalls / count_query,
        'precision': precisions / count_query,
        'hits': hits / count_query
    }

    return results


def jrl_lr_scheduler(optimizer, base_lr, min_lr, current_iter, total_iter):
    new_lr = base_lr * max(min_lr, 1 - current_iter / total_iter)
    optimizer.param_groups[0]['lr'] = new_lr
    return new_lr


def results(rank_list_file, test_qrel_file, rank_cutoff_list):
    qrel_map = {}
    with open(test_qrel_file) as fin:
        for line in fin:
            arr = line.strip().split(' ')
            qid = arr[0]
            did = arr[2]
            label = int(arr[3])
            if label < 1:
                continue
            if qid not in qrel_map:
                qrel_map[qid] = set()
            qrel_map[qid].add(did)

    _results = []
    for rank_cut in rank_cutoff_list:
        _results.append(
            print_metrics_with_rank_cutoff(rank_cut, qrel_map, rank_list_file))
    return _results


#compute ndcg
def metrics(doc_list, rel_set):
    dcg = 0.0
    hit_num = 0.0
    for i in range(len(doc_list)):
        if doc_list[i] in rel_set:
            #dcg
            dcg += 1 / (log(i + 2) / log(2))
            hit_num += 1
    #idcg
    idcg = 0.0
    for i in range(min(len(rel_set), len(doc_list))):
        idcg += 1 / (log(i + 2) / log(2))
    ndcg = dcg / idcg
    recall = hit_num / len(rel_set)
    precision = hit_num / len(doc_list)
    #compute hit_ratio
    hit = 1.0 if hit_num > 0 else 0.0
    large_rel = 1.0 if len(rel_set) > len(doc_list) else 0.0
    return recall, ndcg, hit, large_rel, precision


def check_weight_norm(model: torch.nn.Module):
    for n, param in model.named_parameters():
        if param.grad is not None:
            print(f"{n}: {torch.norm(param.grad)}")
        else:
            print(f"{n}: None")


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print(f"model size {size / 1e3} KB")
    os.remove('temp.p')


def soft_threshold(w, th):
    with torch.no_grad():
        temp = torch.abs(w) - th
        return torch.sign(w) * torch.nn.functional.relu(temp)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class ProgressLogger:

    def __init__(self, cfg, save_dir, is_search):
        self.cfg = cfg
        self.derivation = cfg.MODEL.NAS.DERIVATION
        self.search = is_search
        self.nas = cfg.MODEL.NAS.SELECTION
        self.save_dir = save_dir
        self._log = ddict(list)
        self._metric = ddict(AverageMeter)

    def log_results(self, results):
        self._log['epoch_time'].append(time.time())
        for k, v in results.items():
            self._log[k].append(v)
        return self

    def save(self, name=None):
        df = pd.DataFrame(data=self._log)
        s = 'search' if self.search else 'train'
        if name is None:
            path = os.path.join(self.save_dir,
                                f"{s}_D-{self.derivation}_S-{self.nas}.csv")
        else:
            path = os.path.join(self.save_dir, name)
        df.to_csv(path)
        return self
