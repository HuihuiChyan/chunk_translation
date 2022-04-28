from tqdm import tqdm
import pdb
with open('n_grams.zh', encoding='utf-8') as fn_grams:
	n_grams = [line.strip() for line in fn_grams.readlines()]

with open('test2013.zh.tok.bpe', encoding='utf-8') as ftok,\
open('test2013.zh.chunk', 'w', encoding='utf-8') as fchunk:
	lines = [line.strip().split() for line in ftok.readlines()]
	bpe_lines = []
	for line in tqdm(lines):
		ori_line = line
		chunk_line = ['0' for l in line]
		unbpe2bpe = {-1:-1}
		real_count = 0
		for i,tok in enumerate(line):
			if len(tok)>2:
				if tok[-2:] == '@@':
					continue
			unbpe2bpe[real_count] = i
			real_count+= 1
		line = ' '.join(line).replace('@@ ', '').split()
		assert len(line) == real_count
		for i in range(len(line)):
			for n in [4, 3, 2]:
				if i <= len(line)-n:
					if ' '.join(line[i:i+n]) in n_grams:
						if not '1' in chunk_line[unbpe2bpe[i-1]+1:unbpe2bpe[i+n-1]+1]:
							for pos in range(unbpe2bpe[i-1]+1,unbpe2bpe[i+n-1]+1):
								chunk_line[pos] = '1'
		fchunk.write(' '.join(chunk_line)+'\n')
