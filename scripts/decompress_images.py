import sys
import gzip
import os.path as osp
from array import array
import torch

if __name__ == "__main__":
    _, data_path = sys.argv
    assert osp.isdir(data_path)
    product_ids = []
    with gzip.open(data_path + 'product.txt.gz' , 'rt') as fin:
        for line in fin:
            product_ids.append(line.strip())
    product_size = len(product_ids)
    img_feature_num = 4096
    with open(data_path + 'product_image_feature.b' , 'rb') as fin:
        for i in range(product_size):
            float_array = array('f')
            float_array.fromfile(fin , 4096)
            img_features = torch.tensor(list(float_array))
            torch.save(img_features, osp.join(data_path, f'img_features_{i}.pth.tar'))
            print(f'saving img_features_{i}.pth.tar at {data_path}')
