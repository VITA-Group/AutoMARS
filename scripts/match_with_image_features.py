import gzip
import os
import struct
import sys
from array import array

data_path = sys.argv[1]
image_feature_file = sys.argv[2]

# read needed product ids
product_ids = []
print("program started")
with gzip.open(data_path + 'product.txt.gz', 'r') as fin:
    for line in fin:
        product_ids.append(line.strip())
product_indexes = dict([(product_ids[i], i) for i in range(len(product_ids))])

# read image features
print("read image features")
image_features = [None for i in range(len(product_ids))]
cnt = 0
cnt2 = 0
with open(image_feature_file, 'rb') as f:
    while True:
        product_id = f.read(10)
        if product_id == b'':
            break
        if product_id in product_indexes:
            cnt2 += 1
            if image_features[product_indexes[product_id]] != None:
                print('duplicate ' + product_id)
            feature = []
            for i in range(4096):
                feature.append(struct.unpack('f', f.read(4))[0])
            image_features[product_indexes[product_id]] = feature
        else:
            f.read(4096 * 4)
        print(cnt + 1, product_id, f"{cnt2}/{len(product_ids)}")
        cnt += 1
print("done reading")
count = 0
for i in range(len(product_ids)):
    if image_features[i] == None:
        print('Not found: ', product_ids[i])
        count += 1
print(str(count) + '/' + str(len(product_ids)) + ' do not have image features')

# output image features
# print(" ".join(str(x) for x in image_features[1][500:510]))
with open(data_path + 'product_image_feature.b', 'wb') as fout:
    for feature in image_features:
        if feature == None:
            feature = [0.0 for i in range(4096)]
        float_array = array('f', feature)
        float_array.tofile(fout)

with open(data_path + 'product_image_feature.b', 'rb') as fin:
    float_array = array('f')
    float_array.fromfile(fin, 4096)
    float_array = array('f')
    float_array.fromfile(fin, 4096)
    print(" ".join(str(x) for x in float_array[500:510]))
