# AutoMARS: Searching to Compress Multi-Modality Recommendation Systems 
Code used for AutoMARS: Searching to Compress Multi-Modality Recommendation Systems
# Methodology
![alt text](https://github.com/VITA-Group/AutoMARS/blob/main/thumbnail.JPG)

# Data processing
1. Download Amazon review datasets from http://jmcauley.ucsd.edu/data/amazon/ (e.g. In our paper, we used 5-core data).
2. Stem and remove stop words from the Amazon review datasets if needed (e.g. In our paper, we stem the field of "reviewText" and "summary" without stop words removal)
    1. java -Xmx4g -jar ./jar/AmazonReviewData_preprocess.jar <jsonConfigFile> <review_file> <output_review_file>
        1. <jsonConfigFile>: A json file that specify the file path of stop words list. An example can be found in the root directory. Enter "false" if don’t want to remove stop words.
        2. <review_file>: the path for the original Amazon review data
        3. <output_review_file>: the output path for processed Amazon review data
3. Index datasets
    1. python ./scripts/index_and_filter_review_file.py <review_file> <indexed_data_dir> <min_count>
        1. <review_file>: the file path for the Amazon review data
        2. <indexed_data_dir>: output directory for indexed data
        3. <min_count>: the minimum count for terms. If a term appears less then <min_count> times in the data, it will be ignored.
4. Split train/test
    1. Download the meta data from http://jmcauley.ucsd.edu/data/amazon/
    2. Split datasets for training and test
        1. python ./scripts/split_train_test.py <indexed_data_dir> <review_sample_rate>
        2. <indexed_data_dir>: the directory for indexed data.
        3. <review_sample_rate>: the proportion of reviews used in test for each user (e.g. in our paper, we used 0.3).
5. Match image features
    1. Download the image features from http://jmcauley.ucsd.edu/data/amazon/ .
    2. Match image features with product ids.
        1. python ./scripts/match_with_image_features.py <indexed_data_dir> <image_feature_file>
        2. <indexed_data_dir>: the directory for indexed data.
        3. <image_feature_file>: the file for image features data.
6. Match rating features
    1. Construct latent representations based on rating information with any method you like (e.g. BPR).
    2. Format the latent factors of items and users in "item_factors.csv" and "user_factors.csv" such that each row represents one latent vector for the corresponding item/user in the <indexed_data_dir>/product.txt.gz and user.txt.gz (see the example csv files).
    3. Put the item_factors.csv and user_factors.csv into <indexed_data_dir>
7. Decompress the zip image file into individual images
   1. decompress *.b image feature files into individual image files for dataloading
      1. python ./scripts/decompress_images.py <data_dir>
      2. <data_dir> the folder which contains *.b file after performing step <1-6>. It should be under <indexed_data_dir>/<min_count>/
8. Precompute data distribution
    1. to cache the dataset distribution for later data loading
      1. python ./scripts/process_data.py <data_dir>/<min_count>/
# Running code
```console
# large model
python jrl/main.py --search --train --budget "0.8+0.6" --cfg experiments/<file>.yaml OUTPUT_DIR <output_dir>
# medium model
python jrl/main.py --search --train --budget "0.6+0.4" --cfg experiments/<file>.yaml OUTPUT_DIR <output_dir>
# small model
python jrl/main.py --search --train --budget "0.4+0.2" --cfg experiments/<file>.yaml OUTPUT_DIR <output_dir>
```
# Citation
