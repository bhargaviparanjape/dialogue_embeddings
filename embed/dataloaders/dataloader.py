from abc import ABCMeta
from embed.dataloaders.factory import RegisterBatcher
from collections import defaultdict
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
import random
import pdb
from tqdm import tqdm

class AbstractDataLoader():
	__metaclass__ = ABCMeta

@RegisterBatcher('conversation_length')
class ConversationBatcher(AbstractDataLoader):
	def __init__(self, args):
		self.args = args
		self.K = 4

	def get_batches(self, args, dataset):
		mode = args.run_mode
		batch_size = args.batch_size
		
		def create_conversation_batch(batch_data):
			## all utterances in all conversations of a batch should be masked and each conversation in batch must have same length

			## batchSize * N *
			next_utterance_options_list = []
			gold_ids = []
			conversation_mask = []
			batch = {}
			utterance_list = []
			conversation_lengths = []

			if args.truncate_dataset:
				for idx, conversation in enumerate(batch_data):
					random_short_length = random.sample(range(5,10),1)[0]
					batch_data[idx].utterances = conversation.utterances[:random_short_length]

			max_num_utterances = max([len(c.utterances) for c in batch_data]) ## N + 1(last null utterance has to be there for last actual utterance)
			for c_idx, conversation in enumerate(batch_data):
				length = len(conversation.utterances)
				for u_idx, u in enumerate(conversation.utterances):
					utterance_list.append(u.tokens)
					## randomly sample K items in conversation
					utterance_samples = random.sample(range(length), self.K)
					## last utterance predicts the previous utterance
					if u_idx == length - 1:
						correct_id = u_idx-1
					else:
						correct_id = u_idx+1
					if correct_id not in utterance_samples:
						utterance_samples[random.sample(range(self.K),1)[0]] = correct_id
					np.random.shuffle(utterance_samples)
					gold_index = utterance_samples.index(correct_id)

					utterance_samples = [s + c_idx*max_num_utterances for s in utterance_samples]
					# gold_index += c_idx*max_num_utterances

					next_utterance_options_list.append(utterance_samples)
					gold_ids.append(gold_index)

				conversation_lengths.append(length)
				conversation_mask.append([1]*length + [0]*(max_num_utterances - length))
				for i in range(max_num_utterances - length):
					utterance_list.append([])
					next_utterance_options_list.append([length + i + c_idx*max_num_utterances]*self.K)
					gold_ids.append(0)



			#character_ids = batch_to_ids(utterance_list)
			#dict_ = ee(character_ids)
			#embeddings = dict_['elmo_representations'][0]
			#input_mask = dict_['mask']
			# character_ids = batch_to_id(utterance_list)
			batch['utterance_list'] = utterance_list
			batch['next_utterance_options_list'] = next_utterance_options_list
			batch['next_utterance_gold_ids'] = gold_ids
			batch['conversation_lengths'] = conversation_lengths
			batch['conversation_mask'] = conversation_mask
			## to reshape conversations
			batch['max_num_utterances'] = max_num_utterances
			## TODO: no DA being sent
			return batch

		def create_bucket_batches(dataset):
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
					batch = create_conversation_batch(batch_data)
					batches.append(batch)
			return batches

		if mode == "train":
			train_batches = create_bucket_batches(dataset.train_dataset)
			valid_batches = create_bucket_batches(dataset.valid_dataset)
			test_batches = create_bucket_batches(dataset.test_dataset)
			np.random.shuffle(train_batches)
			return train_batches, valid_batches, test_batches
		elif mode =="test":
			valid_batches = create_bucket_batches(dataset.valid_dataset)
			test_batches = create_bucket_batches(dataset.test_dataset)
			return None,valid_batches,test_batches

