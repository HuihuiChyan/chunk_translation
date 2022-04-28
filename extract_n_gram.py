from tqdm import tqdm
from collections import defaultdict, Counter

zh_puncs = ['，', '。', '？', '！', '；', '、', '：', '；', '‘', '’', '“', '”', '—', '——']
en_puncs = [',', '.', '?', '!', ';', ':', ';', "'", '"', '-']
puncs = set(zh_puncs + en_puncs)

def extract_n_gram(lines, n, top_k):
	# 抽出top_k，大小为n的N-gram
	n_gram_candis = []
	for line in tqdm(lines):
		for i in range(len(line)-n+1):
			if all([l not in puncs for l in line[i:i+n]]):
				n_gram = ' '.join(line[i:i+n])
				n_gram_candis.append(n_gram)
	counter = Counter(n_gram_candis)
	n_grams = [line[0] for line in counter.most_common(top_k)]
	# 这里你也可以设置一个阈值，把出现次数小于某个值的N-gram都去掉
	return n_grams

with open('train.zh.extract.tok', 'r', encoding='utf-8') as fzh:
	lines = [line.strip().split() for line in fzh.readlines()]
	
	two_grams = extract_n_gram(lines, 2, 1000)
	three_grams = extract_n_gram(lines, 3, 1000)
	four_grams = extract_n_gram(lines, 4, 1000)

with open('n_grams.zh', 'w', encoding='utf-8') as fn_gram:
	n_grams = two_grams + three_grams + four_grams
	for n_gram in n_grams:
		fn_gram.write(n_gram+'\n')