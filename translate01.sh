# sacremoses tokenize -x < demo_data/train.en.extract > demo_data/train.en.extract.tok
# python -m jieba -d ' ' demo_data/train.zh.extract > demo_data/train.zh.extract.tok

# subword-nmt learn-bpe -s 32000 < demo_data/train.en.extract.tok > demo_data/bpe_code.en
# subword-nmt learn-bpe -s 32000 < demo_data/train.zh.extract.tok > demo_data/bpe_code.zh
# subword-nmt apply-bpe -c demo_data/bpe_code.en < demo_data/train.en.extract.tok > demo_data/train.en.extract.tok.bpe
# subword-nmt apply-bpe -c demo_data/bpe_code.zh < demo_data/train.zh.extract.tok > demo_data/train.zh.extract.tok.bpe

# sacremoses tokenize -x < demo_data/test2013.en > demo_data/test2013.en.tok
# python -m jieba -d ' ' demo_data/test2013.zh > demo_data/test2013.zh.tok
# subword-nmt apply-bpe -c demo_data/bpe_code.en < demo_data/test2013.en.tok > demo_data/test2013.en.tok.bpe
# subword-nmt apply-bpe -c demo_data/bpe_code.zh < demo_data/test2013.zh.tok > demo_data/test2013.zh.tok.bpe

# sacremoses tokenize -x < demo_data/test2015.en > demo_data/test2015.en.tok
# python -m jieba -d ' ' demo_data/test2015.zh > demo_data/test2015.zh.tok
# subword-nmt apply-bpe -c demo_data/bpe_code.en < demo_data/test2015.en.tok > demo_data/test2015.en.tok.bpe
# subword-nmt apply-bpe -c demo_data/bpe_code.zh < demo_data/test2015.zh.tok > demo_data/test2015.zh.tok.bpe

# mkdir demo_data/data-bin
# cp demo_data/train.en.extract.tok.bpe demo_data/data-bin/train.en
# cp demo_data/train.zh.extract.tok.bpe demo_data/data-bin/train.zh
# cp demo_data/test2013.en.tok.bpe demo_data/data-bin/dev.en
# cp demo_data/test2013.zh.tok.bpe demo_data/data-bin/dev.zh

# fairseq-preprocess \
# 	--source-lang en \
# 	--target-lang zh \
# 	--trainpref demo_data/data-bin/train \
# 	--validpref demo_data/data-bin/dev \
# 	--destdir demo_data/data-bin \
# 	--thresholdsrc 2 \
# 	--thresholdtgt 2
 
export CUDA_VISIBLE_DEVICES=6
fairseq-train demo_data/data-bin \
	--user-dir user_dir \
	--arch chunk_translation_model_base \
	--task chunk_translation_task \
	--add_crf_loss \
	--share-decoder-input-output-embed \
	--clip-norm 0.0 \
	--max-tokens 4096 \
	--lr 2e-4 \
	--lr-scheduler inverse_sqrt \
	--warmup-updates 5000 \
	--optimizer adam \
	--adam-betas '(0.9, 0.98)' \
	--dropout 0.3 \
	--weight-decay 0.0001 \
	--criterion chunk_translation_criterion \
	--label-smoothing 0.1 \
	--eval-bleu \
	--eval-bleu-args '{"beam":5, "max_len_a": 1.2, "max_len_b": 10}' \
	--eval-bleu-remove-bpe \
	--eval-bleu-print-samples \
	--best-checkpoint-metric bleu \
	--maximize-best-checkpoint-metric \
	--save-dir checkpoints/en2zh01 \
	--patience 5 \
	--add_crf_loss \
	--max-tokens-valid 1024 

# fairseq-interactive demo_data/data-bin \
# 	-s en \
# 	-t zh \
#	 --user-dir user_dir \
# 	--path checkpoints/en2zh03/checkpoint_best.pt \
# 	--beam 5 \
# 	--batch-size 200 \
# 	--buffer-size 200 \
# 	--no-progress-bar \
# 	--unkpen 5 < demo_data/test2015.en.tok.bpe
 
# grep -a ^H ru2zh_data/data-bin/dev.zh.out | cut -f3- > ru2zh_data/data-bin/dev.zh.out.grep