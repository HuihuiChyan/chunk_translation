import os
import math
import torch
import logging
import fairseq
import itertools
import numpy as np

from torch import Tensor
from typing import Dict, List, Optional
from fairseq.data import data_utils, indexed_dataset, FairseqDataset
from fairseq.tasks import register_task
from fairseq.models import register_model, register_model_architecture
from fairseq.criterions import register_criterion

from fairseq.tasks.translation import TranslationTask
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.models.transformer import TransformerModel, base_architecture
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss

import pdb

logger = logging.getLogger(__name__)

def collate(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False, input_feeding=True):

	if len(samples) == 0:
		return {}

	def merge(key, left_pad, move_eos_to_beginning=False):
		return data_utils.collate_tokens([s[key] for s in samples], pad_idx, eos_idx, left_pad, move_eos_to_beginning)

	id = torch.LongTensor([s["id"] for s in samples])
	src_tokens = merge("source", left_pad=left_pad_source)

	# 按照递减的长度进行排序
	src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])
	src_lengths, sort_order = src_lengths.sort(descending=True)
	id = id.index_select(0, sort_order)
	src_tokens = src_tokens.index_select(0, sort_order)

	prev_output_tokens = None
	target = None

	target = merge("target", left_pad=left_pad_target)
	target = target.index_select(0, sort_order)
	tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples]).index_select(0, sort_order)
	ntokens = tgt_lengths.sum().item()

	chunk_line = [torch.LongTensor(s['chunk_line']+[2]) for s in samples]
	chunk = data_utils.collate_tokens(chunk_line, pad_idx=3, left_pad=left_pad_target)
	chunk = chunk.index_select(0, sort_order)

	if input_feeding:
		prev_output_tokens = merge("target", left_pad=left_pad_target, move_eos_to_beginning=True)
		prev_chunk = data_utils.collate_tokens(chunk_line, pad_idx=3, eos_idx=2, left_pad=left_pad_target, move_eos_to_beginning=True)

	batch = {
		"id":id,
		"nsentences": len(samples),
		"ntokens": ntokens,
		"net_input":{
			"src_tokens": src_tokens,
			"src_lengths": src_lengths,
		},
		"target": target,
		"chunk": chunk
	}

	if prev_output_tokens is not None:
		batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(0, sort_order)
		batch["prev_chunk"] = prev_chunk.index_select(0, sort_order)

	return batch

class ChunkDynamicCRF(fairseq.modules.DynamicCRF):

    def _compute_chunk_score(self, scores, targets, masks=None):
        batch_size, seq_len = targets.size()
        emission_scores = scores
        transition_scores = (self.E1(targets[:, :-1]) * self.E2(targets[:, 1:])).sum(2)

        scores = emission_scores
        scores[:, 1:] += transition_scores

        if masks is not None:
            scores = scores * masks.type_as(scores)
        return scores.sum(-1)

class ChunkTranslationDataset(LanguagePairDataset):

	def __init__(
		self,
		src,
		src_sizes,
		src_dict,
		tgt,
		tgt_sizes,
		tgt_dict,
		left_pad_source=True,
		left_pad_target=False,
		shuffle=True,
		input_feeding=True,
	):
		super(ChunkTranslationDataset, self).__init__(
			src,
			src_sizes,
			src_dict,
			tgt,
			tgt_sizes,
			tgt_dict,
			left_pad_source=True,
			left_pad_target=False,
			shuffle=True,
			input_feeding=True
		)
		if shuffle:
			chunk_file = 'demo_data/train.zh.chunk'
		else:
			chunk_file = 'demo_data/test2013.zh.chunk'
		with open(chunk_file, encoding='utf-8') as fchunk:
			chunk_lines = [[int(l) for l in line.strip().split()] for line in fchunk.readlines()]
		self.chunk_lines = fairseq.data.RawLabelDataset(chunk_lines)

	def __getitem__(self, index):
		tgt_item = self.tgt[index]
		src_item = self.src[index]
		chunk_line = self.chunk_lines[index]
		example = {"id": index, "source": src_item, "target": tgt_item, "chunk_line": chunk_line}
		return example

	def collater(self, samples):
		res = collate(
			samples,
			pad_idx=self.src_dict.pad(),
			eos_idx=self.eos,
			left_pad_source=self.left_pad_source,
			left_pad_target=self.left_pad_target,
			input_feeding=self.input_feeding)
		return res

def load_chunk_langpair_dataset(
	data_path,
	split,
	src,
	src_dict,
	tgt,
	tgt_dict,
	dataset_impl,
	left_pad_source,
	left_pad_target, 
	shuffle=True
):

	def split_exists(split, src, tgt, lang, data_path):
		filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
		return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

	src_datasets = []
	tgt_datasets = []
	
	for k in itertools.count():
		split_k = split + (str(k) if k > 0 else "")

		if split_exists(split_k, src, tgt, src, data_path):
			prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
		elif k > 0:
			break
		else:
			raise FileNotFoundError("Dataset not fount: {} ({})".format(split, data_path))

		src_datasets.append(data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl))
		tgt_datasets.append(data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl))

		logger.info("{} {} {}-{} {} examples".format(data_path, split_k, src, tgt, len(src_datasets[-1])))

	assert len(src_datasets) == len(tgt_datasets)

	if len(src_datasets) == 1:
		src_dataset = src_datasets[0]
		tgt_dataset = tgt_datasets[0]
	else:
		sample_ratios = [1] * len(src_datasets)
		src_dataset = ConcatDataset(src_datasets, sample_ratios)
		tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

	return ChunkTranslationDataset(src_dataset,
								   src_dataset.sizes,
								   src_dict,
								   tgt_dataset,
								   tgt_dataset.sizes,
								   tgt_dict,
								   shuffle=shuffle)

class ChunkSequenceGenerator(fairseq.sequence_generator.SequenceGenerator):

	def _generate(
		self,
		sample: Dict[str, Dict[str, Tensor]],
		prefix_tokens: Optional[Tensor] = None,
		constraints: Optional[Tensor] = None,
		bos_token: Optional[int] = None,
	):

		incremental_states = [{} for i in range(self.model.models_size)]
		src_tokens = sample["net_input"]["src_tokens"]
		src_lengths = ((src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1))

		# bsz: total number of sentences in beam
		bsz, src_len = src_tokens.size()[:2]
		beam_size = self.beam_size
		num_remaining_sent = bsz  # number of sentences remaining
		cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

		max_len = min(int(self.max_len_a * src_len + self.max_len_b), self.model.max_decoder_positions() - 1)
		assert (self.min_len <= max_len), "min_len cannot be larger than max_len, please adjust these!"
		encoder_outs = self.model.forward_encoder(sample["net_input"])

		# placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
		new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1).to(src_tokens.device).long()
		encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
		assert encoder_outs is not None

		# initialize buffers
		scores = (torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float())
		# beamed_lprobs = (torch.zeros(bsz * beam_size, max_len + 1, 25).to(src_tokens).float())
		# +1 for eos; pad is never chosen for scoring
		tokens = (torch.zeros(bsz * beam_size, max_len + 2).to(src_tokens).long().fill_(self.pad))  
		# +2 for eos and pad
		tokens[:, 0] = self.eos
		attn: Optional[Tensor] = None

		finalized = [[] for i in range(bsz)]
		# contains lists of dictionaries of infomation about the hypothesis being finalized at each step

		finished = [False for i in range(bsz)]  
		# a boolean array indicating if the sentence at the index is finished or not

		# offset arrays for converting between different indexing schemes
		bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
		cand_offsets = torch.arange(0, cand_size).type_as(tokens)

		reorder_state: Optional[Tensor] = None
		batch_idxs: Optional[Tensor] = None

		original_batch_idxs: Optional[Tensor] = None
		if "id" in sample and isinstance(sample["id"], Tensor):
			original_batch_idxs = sample["id"]
		else:
			original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

		for step in range(max_len + 1):  # one extra step for EOS marker
			# reorder decoder internal states based on the prev choice of beams
			# print(f'step: {step}')
			if reorder_state is not None:
				if batch_idxs is not None:
					# update beam indices to take into account removed sentences
					corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
					reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
					original_batch_idxs = original_batch_idxs[batch_idxs]

				self.model.reorder_incremental_state(incremental_states, reorder_state)
				encoder_outs = self.model.reorder_encoder_out(encoder_outs, reorder_state)

			lprobs, _ = self.model.forward_decoder(tokens[:, : step + 1], encoder_outs,
												   incremental_states, self.temperature)
			lprobs[:, self.pad] = -math.inf  # never select pad
			lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty	
			if step >= max_len: # handle max length constraint
				lprobs[:, : self.eos] = -math.inf
				lprobs[:, self.eos + 1 :] = -math.inf
			if step < self.min_len: # (does not apply if using prefix_tokens)
				lprobs[:, self.eos] = -math.inf

			# Shape: (batch, cand_size)
			cand_scores, cand_indices, cand_beams = self.search.step(
				step, lprobs.view(bsz, -1, self.vocab_size),
				scores.view(bsz, beam_size, -1)[:, :, :step],
				tokens[:, : step + 1], original_batch_idxs,
			)

			cand_bbsz_idx = cand_beams.add(bbsz_offsets)
			lprobs = lprobs.topk(25, dim=-1).values
			if step == 0:
				cumulative_lprobs = lprobs.unsqueeze(-2)
			else:
				cumulative_lprobs = torch.cat((cumulative_lprobs, lprobs.unsqueeze(-2)), -2)

			# pdb.set_trace()
			# cand_cumulative_lprobs = torch.index_select(cumulative_lprobs, 0, cand_beams.view(-1)).view(bsz, cand_size, -1, 25)

			eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)

			# only consider eos when it's among the top beam_size indices.
			eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])

			finalized_sents: List[int] = []
			if eos_bbsz_idx.numel() > 0:			
				eos_scores = torch.masked_select(cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size])
				finalized_sents = self.finalize_hypos(step, eos_bbsz_idx, eos_scores, tokens, scores, finalized,
													  finished, beam_size, None, src_lengths, max_len, cumulative_lprobs)

				num_remaining_sent -= len(finalized_sents)

			assert num_remaining_sent >= 0
			if num_remaining_sent == 0:
				break
			if self.search.stop_on_max_len and step >= max_len:
				break
			assert step < max_len

			if len(finalized_sents) > 0:
				new_bsz = bsz - len(finalized_sents)

				# construct batch_idxs which holds indices of batches to keep for the next pass
				batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
				batch_mask[finalized_sents] = False
				batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(batch_mask)

				eos_mask = eos_mask[batch_idxs]; cand_beams = cand_beams[batch_idxs]
				bbsz_offsets.resize_(new_bsz, 1); cand_bbsz_idx = cand_beams.add(bbsz_offsets)
				cand_scores = cand_scores[batch_idxs]; cand_indices = cand_indices[batch_idxs]
				src_lengths = src_lengths[batch_idxs]; cand_cumulative_lprobs = cand_cumulative_lprobs[batch_idxs]
				
				scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
				tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)

				bsz = new_bsz
			else:
				batch_idxs = None

			# Set active_mask so that values > cand_size indicate eos hypos and values < cand_size indicate candidate active hypos.
			# After, the min values per row are the top candidate active hypos.
			# Then get the top beam_size active hypotheses, which are just the hypos with the smallest values in active_mask. 
			# {active_hypos} indicates which {beam_size} hypotheses from the list of {2 * beam_size} candidates were selected. 

			# Shapes: (batch size, beam size)
			active_mask = torch.add(eos_mask.type_as(cand_offsets) * cand_size, cand_offsets[: eos_mask.size(1)])
			active_hypos = torch.topk(active_mask, k=beam_size, dim=1, largest=False).indices

			# {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam can be selected more than once).
			# Copy tokens and scores for active hypotheses. Set the tokens for each beam (can select the same row more than once).
			active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos).view(-1)
			tokens[:, : step + 1] = torch.index_select(tokens[:, : step + 1], dim=0, index=active_bbsz_idx)
			if step > 0:
				scores[:, : step] = torch.index_select(scores[:, : step], dim=0, index=active_bbsz_idx)

			tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(cand_indices, dim=1, index=active_hypos)
			scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(cand_scores, dim=1, index=active_hypos)

			# reorder incremental state in decoder
			reorder_state = active_bbsz_idx

		tgt_tokens = []
		# sort by score descending
		for sent in range(len(finalized)):
			tgt_tokens = torch.tensor([elem["tokens"].tolist() for elem in finalized[sent]]).cuda()
			tgt_lprobs = torch.tensor([elem['cumulative_lprobs'].tolist() for elem in finalized[sent]]).cuda()
			tgt_scores = torch.tensor([float(elem['score'].item()) for elem in finalized[sent]]).cuda()	
			# log_scores = torch.tensor([float(elem["score"].item()) for elem in finalized[sent]])
			pos_scores = torch.tensor([elem["positional_scores"].tolist() for elem in finalized[sent]]).cuda()
			crf_scores = self.model.single_model.dynamic_crf_layer._compute_chunk_score(pos_scores, tgt_tokens)
			scores = tgt_scores #+ crf_scores * 0.01
			# print(tgt_scores.mean())
			# print(crf_scores.mean())
			_, sorted_scores_indices = torch.sort(scores, descending=True)

			finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]

		return finalized


@register_task('chunk_translation_task')
class ChunkTranslationTask(TranslationTask):

	@staticmethod
	def add_args(parser):
		TranslationTask.add_args(parser)
		parser.add_argument("--add_n_gram_loss", default=False, action="store_true")
		parser.add_argument("--add_crf_loss", default=False, action="store_true")
		parser.add_argument("--crf_loss_ratio", default=0.1, type=float)

	def load_dataset(self, split, epoch=1, **kwargs):
		paths = fairseq.utils.split_paths(self.args.data)
		data_path = paths[(epoch -1) % len(paths)]
		src, tgt = self.args.source_lang, self.args.target_lang
		self.datasets[split] = load_chunk_langpair_dataset(
			data_path,
			split,
			src,
			self.src_dict,
			tgt,
			self.tgt_dict,
			dataset_impl=self.args.dataset_impl,
			left_pad_source=self.args.left_pad_source,
			left_pad_target=self.args.left_pad_target,
			shuffle=(split == "train"),
		)

	def build_generator(self, models, args):
		if getattr(args, "score_reference", False):
			from fairseq.sequence_scorer import SequenceScorer
			return SequenceScorer(self.target_dictionary, compute_alignment=False)

		search_strategy = fairseq.search.BeamSearch(self.target_dictionary)

		if self.args.add_crf_loss:
			return ChunkSequenceGenerator(models, self.target_dictionary, beam_size=getattr(args, 'beam_size', 5), search_strategy=search_strategy)
		else:
			return fairseq.sequence_generator.SequenceGenerator(models, self.target_dictionary,
																beam_size=getattr(args, 'beam_size', 5),
																search_strategy=search_strategy)
@register_model("chunk_translation_model")
class ChunkTranslationModel(TransformerModel):

	def __init__(self, args, encoder, decoder, chunk_predictor, dynamic_crf_layer):
		super().__init__(args, encoder, decoder)
		self.args = args
		self.supports_align_args = True
		self.chunk_predictor = chunk_predictor
		self.dynamic_crf_layer = dynamic_crf_layer

	@staticmethod
	def add_args(parser):
		super(ChunkTranslationModel, ChunkTranslationModel).add_args(parser)
		parser.add_argument("--crf_rerank_ratio", default=0.3, type=float)
		parser.add_argument("--combined_rerank", default=False, action="store_true")

	@classmethod
	def build_model(cls, args, task):

		chunk_translation_model_base(args)

		src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

		encoder_embed_tokens = cls.build_embedding(args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
		decoder_embed_tokens = cls.build_embedding(args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)

		encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
		decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

		chunk_predictor = torch.nn.Linear(args.decoder_output_dim, 3)

		dynamic_crf_layer = ChunkDynamicCRF(len(task.target_dictionary), low_rank=32, beam_size=64)
		return cls(args, encoder, decoder, chunk_predictor, dynamic_crf_layer)

chunk_translation_model_base = base_architecture
register_model_architecture("chunk_translation_model", "chunk_translation_model_base")(chunk_translation_model_base)

@register_criterion('chunk_translation_criterion')
class ChunkTranslationCriterion(LabelSmoothedCrossEntropyCriterion):

	def compute_chunk_loss(self, chunk_predictions, chunk_labels, reduce=True):
		chunk_predictions = chunk_predictions.permute([1,0,2]).contiguous().view(-1, chunk_predictions.size(-1))
		chunk_labels = chunk_labels.view(-1)
		loss = torch.nn.functional.cross_entropy(chunk_predictions, chunk_labels, ignore_index=3, reduce=reduce)
		return loss

	def forward(self, model, sample, reduce=True):
		net_output = model(**sample["net_input"])
		last_inner_states = net_output[1]['inner_states'][-1]
		target_padding_mask = sample['target'].ne(model.decoder.padding_idx)		
		token_loss, token_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)	
		loss = token_loss

		if self.task.args.add_n_gram_loss:
			chunk_predictions = model.chunk_predictor(last_inner_states)
			chunk_loss = self.compute_chunk_loss(chunk_predictions, sample['chunk'], reduce=reduce)
			loss += chunk_loss

		if self.task.args.add_crf_loss:
			crf_log = model.dynamic_crf_layer(net_output[0], sample["target"], target_padding_mask)
			loss -= self.task.args.crf_loss_ratio * crf_log.sum(-1)

		sample_size = sample["ntokens"]
		
		logging_output = {
			"token_loss": token_loss.data,
			"token_nll_loss": token_nll_loss.data,
			"ntokens": sample["ntokens"],
			"nsentences": sample["target"].size(0),
			"sample_size": sample_size,
		}

		return loss, sample_size, logging_output