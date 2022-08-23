import os,sys
from math import log

rank_list_file = sys.argv[1]
test_qrel_file = sys.argv[2]
rank_cutoff_list = [int(cut) for cut in sys.argv[3].split(',')]

# rank_list_file = './results_official_split/reviews_Beauty_5.json.gz.stem.nostop/simplified_bpr_text_image/emb300-300/test.product.ranklist'
# test_qrel_file = '/hdd3/haotao/CIKM2017/reviews_Beauty_5.json.gz.stem.nostop/min_count5/query_split/test.qrels'
# rank_cutoff_list = [10]

# rank_list_file = './results/reviews_Beauty_5/simplified_bpr_text_image/emb300-300/test.product.ranklist'
# test_qrel_file = '/hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/test.qrels'
# rank_cutoff_list = [10]

#read qrel file
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

#compute ndcg
def metrics(doc_list, rel_set):
	dcg = 0.0
	hit_num = 0.0
	for i in range(len(doc_list)):
		if doc_list[i] in rel_set:
			#dcg
			dcg += 1/(log(i+2)/log(2))
			hit_num += 1
	#idcg
	idcg = 0.0
	for i in range(min(len(rel_set),len(doc_list))):
		idcg += 1/(log(i+2)/log(2))
	ndcg = dcg/idcg
	recall = hit_num / len(rel_set)
	precision = hit_num / len(doc_list)
	#compute hit_ratio
	hit = 1.0 if hit_num > 0 else 0.0
	large_rel = 1.0 if len(rel_set) > len(doc_list) else 0.0
	return recall, ndcg, hit, large_rel, precision

def print_metrics_with_rank_cutoff(rank_cutoff):
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
			recall, ndcg, hit, large_rel, precision = metrics(rank_list[qid],qrel_map[qid])
			count_query += 1
			ndcgs += ndcg
			recalls += recall
			hits += hit
			large_rels += large_rel
			precisions += precision

	print("Query Number:" + str(count_query))
	print("Larger_rel_set@"+str(rank_cutoff) + ":" + str(large_rels/count_query))
	print("Recall@"+str(rank_cutoff) + ":" + str(recalls/count_query))
	print("Precision@"+str(rank_cutoff) + ":" + str(precisions/count_query))
	print("NDCG@"+str(rank_cutoff) + ":" + str(ndcgs/count_query))
	print("Hit@"+str(rank_cutoff) + ":" + str(hits/count_query))

for rank_cut in rank_cutoff_list:
	print_metrics_with_rank_cutoff(rank_cut)
