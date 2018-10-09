from abc import ABCMeta
from src.dataloaders.factory import RegisterBatcher
from collections import defaultdict
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
import random
import pdb
from tqdm import tqdm
from src.utils.utility_functions import pad_seq

class AbstractDataLoader():
	__metaclass__ = ABCMeta


@RegisterBatcher('conversation_snippet')
class ConversationSnippetBatcher(AbstractDataLoader):
	def __init__(self, args):
		self.args = args

## all utterances in all conversations of a batch should be masked and each conversation in batch must have same length
@RegisterBatcher('conversation_length')
class ConversationBatcher(AbstractDataLoader):
	def __init__(self, args):
		self.args = args
		self.K = args.K

	def get_batches(self, args, dataset):
		mode = args.run_mode
		batch_size = args.batch_size
		
		def create_conversation_batch(batch_data, vocabulary):

			## for classification
			utterance_options_list = []
			next_gold_ids = []
			prev_gold_ids = []

			## utterance vocabulary ids for next/previous utterance reconstruction (bow model and decoding)
			next_utterance_ids_list = []
			previous_utterance_ids_list = []
			next_utterance_bow_list = []
			prev_utterance_bow_list= []

			## category classification labels
			labels = []

			conversation_mask = []
			utterance_list = []
			utterance_ids_list = []
			utterance_word_ids_list = []
			conversation_lengths = []
			conversation_ids = []

			batch = {}

			if args.truncate_dataset:
				for idx, conversation in enumerate(batch_data):
					random_short_length = random.sample(range(5,10),1)[0]
					batch_data[idx].utterances = conversation.utterances[:random_short_length]

			max_num_utterances = max([len(c.utterances) for c in batch_data])
			max_utterance_length = max([max([len(u.tokens) for u in c.utterances]) for c in batch_data])
			for c_idx, conversation in enumerate(batch_data):
				length = len(conversation.utterances)
				for u_idx, u in enumerate(conversation.utterances):
					utterance_list.append(u.tokens)
					utterance_word_ids_list.append(pad_seq(vocabulary.get_indices(u.tokens), max_utterance_length))
					utterance_ids_list.append(u.id)
					labels.append(u.label)
					## randomly sample K items in conversation
					utterance_samples = random.sample(range(length), self.K)

					## last utterance predicts the previous utterance and first utterance predicts next utterance
					if u_idx == length - 1:
						next_id = u_idx-1
						prev_id = u_idx-1
						num_samples = self.K - 1
					elif u_idx == 0:
						next_id = u_idx + 1
						prev_id = u_idx + 1
						num_samples = self.K - 1
					else:
						next_id = u_idx+1
						prev_id = u_idx-1
						num_samples = self.K - 2
					next_utterance_ids = vocabulary.get_indices(conversation.utterances[next_id].tokens)
					previous_utterance_ids = vocabulary.get_indices(conversation.utterances[prev_id].tokens)
					current_utterance_ids = vocabulary.get_indices(conversation.utterances[u_idx].tokens)
					prev_utterance_bow = list(set(previous_utterance_ids))
					next_utterance_bow = list(set(next_utterance_ids))

					## randomly sample K items in conversation
					options = list(set(range(length)) - set([next_id, prev_id]))
					utterance_samples = np.random.choice(options, num_samples, replace=False)
					utterance_samples = np.concatenate((utterance_samples, list(set([prev_id,next_id]))), axis=0)
					np.random.shuffle(utterance_samples)
					gold_next_index = np.where(utterance_samples==next_id)[0][0]
					gold_prev_index = np.where(utterance_samples==prev_id)[0][0]

					# for easy indexing when utterances of a batch are in one contiguous list
					utterance_samples = [sample + c_idx*max_num_utterances for sample in utterance_samples]
					utterance_options_list.append(utterance_samples)
					next_gold_ids.append(gold_next_index)
					prev_gold_ids.append(gold_prev_index)
					next_utterance_ids_list.append(next_utterance_ids)
					previous_utterance_ids_list.append(previous_utterance_ids)
					next_utterance_bow_list.append(next_utterance_bow)
					prev_utterance_bow_list.append(prev_utterance_bow)


				conversation_lengths.append(length)
				conversation_ids.append(conversation.id)
				conversation_mask.append([1]*length + [0]*(max_num_utterances - length))
				## apend dummy utterances
				for i in range(max_num_utterances - length):
					utterance_list.append([])
					utterance_word_ids_list.append(pad_seq([], max_utterance_length))
					utterance_ids_list.append(-1)
					utterance_options_list.append([length + i + c_idx*max_num_utterances]*self.K)
					next_gold_ids.append(0)
					prev_gold_ids.append(0)
					labels.append(0)


			batch['utterance_list'] = utterance_list
			batch['utterance_word_ids'] = np.array(utterance_word_ids_list)
			batch['utterance_ids_list'] = np.array(utterance_ids_list)
			batch['input_mask'] = (1*(batch['utterance_word_ids']!= 0))

			batch['utterance_options_list'] = utterance_options_list
			batch['next_utterance_gold'] = next_gold_ids
			batch['next_utterance_ids'] = next_utterance_ids_list
			batch['prev_utterance_gold'] = prev_gold_ids
			batch['prev_utterance_ids'] = previous_utterance_ids_list
			batch['label'] = labels

			batch['conversation_lengths'] = conversation_lengths
			batch['conversation_ids'] = conversation_ids
			batch['conversation_mask'] = conversation_mask
			batch['max_num_utterances'] = max_num_utterances
			batch['max_utterance_length'] = max_utterance_length

			return batch

		def create_bucket_batches(dataset, vocabulary):
			batches = []
			buckets = defaultdict(list)
			for data_item in dataset:
				buckets[data_item.length].append(data_item)
			for src_len in buckets:
				bucket = buckets[src_len]
				np.random.shuffle(bucket)
				num_batches = int(np.ceil(len(bucket) * 1.0 / batch_size))
				for i in range(num_batches):
					cur_batch_size = batch_size if i < num_batches - 1 else len(bucket) - batch_size * i
					begin_index = i * batch_size
					end_index = begin_index + cur_batch_size
					batch_data = list(bucket[begin_index:end_index])
					batch = create_conversation_batch(batch_data, vocabulary)
					batches.append(batch)
			return batches

		if mode == "train":
			train_batches = create_bucket_batches(dataset.train_dataset, dataset.vocabulary)
			valid_batches = create_bucket_batches(dataset.valid_dataset, dataset.vocabulary)
			test_batches = create_bucket_batches(dataset.test_dataset, dataset.vocabulary)
			np.random.shuffle(train_batches)
			return train_batches, valid_batches, test_batches
		elif mode =="test":
			valid_batches = create_bucket_batches(dataset.valid_dataset, dataset.vocabulary)
			test_batches = create_bucket_batches(dataset.test_dataset, dataset.vocabulary)
			return None,valid_batches,test_batches

