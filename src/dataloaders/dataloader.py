from abc import ABCMeta
from src.dataloaders.AbstractDataset import AbstractDataset
from src.dataloaders.factory import RegisterBatcher, RegisterLoader
from collections import defaultdict
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
import random
import pdb
from tqdm import tqdm
from src.utils.utility_functions import pad_seq
from torch.utils.data.sampler import Sampler
import torch
import copy

class AbstractDataLoader():
	__metaclass__ = ABCMeta

class SingleSnippetSampler(Sampler):
	def __init__(self, data, batch_size, shuffle=True):
		self.id_snippet_dict = data[0]
		self.lengths = np.array(data[1])
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.num_snippets = 10

	def __iter__(self):
		chosen_snippets = []
		# Sample at most num_snippets from each conversation
		for conversation_id in self.id_snippet_dict:
			snippet_choices = self.id_snippet_dict[conversation_id]
			if len(snippet_choices) < self.num_snippets:
				chosen_snippets += snippet_choices
			else:
				sampled_snippets = np.random.choice(snippet_choices, self.num_snippets, replace=False)
				chosen_snippets += sampled_snippets.tolist()
		# Index lengths of the chosen snippets
		chosen_lengths = self.lengths[chosen_snippets]

		# Chosen snippets will no longer be contiguous
		lengths = np.array(
			[(-l, np.random.random()) for l in chosen_lengths],
			dtype=[('l', np.int_), ('rand', np.float_)]
		)
		# Indices will be contiguous, you want their corresponding values in chosen_snippets
		indices = np.argsort(lengths, order=('l', 'rand'))
		batches = [[chosen_snippets[j] for j in indices[i:i + self.batch_size]]
		           for i in range(0, len(indices), self.batch_size)]
		if self.shuffle:
			np.random.shuffle(batches)
		return iter([i for batch in batches for i in batch])



	def __len__(self):
		return len(self.keys) * self.num_snippets

class SortedBatchSampler(Sampler):
	def __init__(self, lengths, batch_size, shuffle=True):
		self.lengths = lengths
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.num_snippets = 5

	def __iter__(self):
		lengths = np.array(
			[(-l, np.random.random()) for l in self.lengths],
			dtype=[('l', np.int_), ('rand', np.float_)]
		)
		indices = np.argsort(lengths, order=('l', 'rand'))
		batches = [indices[i:i + self.batch_size]
				   for i in range(0, len(indices), self.batch_size)]
		if self.shuffle:
			np.random.shuffle(batches)
		return iter([i for batch in batches for i in batch])

	def __len__(self):
		return len(self.lengths)

@RegisterLoader('conversation_snippets')
class ConversationSnippetDataloader(AbstractDataLoader):
	def __init__(self, args):
		self.args = args
		self.mode = args.run_mode
		self.batch_size = args.batch_size
		self.conversation_size = args.conversation_size

	def get_dataloader(self, args, dataset, model):

		def create_snippets(dataset, vocabulary, model, mode="train"):

			if args.truncate_dataset:
				for idx, conversation in enumerate(dataset):
					random_short_length = random.sample(range(15,30),1)[0]
					dataset[idx].utterances = conversation.utterances[:random_short_length]

			snippet_bucket = []
			snippet_bucket_length = 0
			snippet_lengths = []
			id_snippet_dict = {}
			for c_idx, conversation in enumerate(dataset):
				length = len(conversation.utterances)
				if hasattr(model.token_encoder, 'conversation_id2idx'):
					conversation_id = model.token_encoder.conversation_id2idx[conversation.id]
				else:
					conversation_id = conversation.id
				snippets = []
				dummy_utterance = [AbstractDataset.Utterance([vocabulary.pad_token])]
				dummy_utterance[0].ids = [0]
				start_utterance = [AbstractDataset.Utterance([vocabulary.soc])]
				end_utterance = [AbstractDataset.Utterance([vocabulary.eoc])]
				if length <= self.conversation_size:
					snippet = {}
					snippet["id"] = conversation_id
					for u_, u in enumerate(conversation.utterances):
						conversation.utterances[u_].ids = vocabulary.get_indices(u.tokens)
					snippet["utterances"] = conversation.utterances #end_utterance
					snippet["range"] = list(range(0, length))
					snippet["mask"] = [1]*(length)
					snippet["length"] = len(snippet["utterances"])
					snippets.append(snippet)
					snippet_lengths.append(snippet["length"])
				else:
					# Pad with dummpy utterances on both sides
					padded_dummy_utterances = dummy_utterance*(self.conversation_size-1)
					for u_, u in enumerate(conversation.utterances):
						conversation.utterances[u_].ids = vocabulary.get_indices(u.tokens)
					padded_utterances = padded_dummy_utterances + conversation.utterances + padded_dummy_utterances
					padded_range = [-1]*(self.conversation_size-1) + list(range(length)) + [-1]*(self.conversation_size-1)
					for i in range(length + self.conversation_size - 1):
						snippet = {}
						snippet["id"] = conversation_id
						snippet_utterences = padded_utterances[i : i+ self.conversation_size]
						snippet_range = padded_range[i : i+ self.conversation_size]
						## Pull Data to the front of the snippet
						if i < self.conversation_size - 1:
							snippet_utterences = snippet_utterences[self.conversation_size-i-1:] + snippet_utterences[0:self.conversation_size-i-1]
							snippet_range = snippet_range[self.conversation_size-i-1:] + snippet_range[0:self.conversation_size-i-1]
						snippet["utterances"] = snippet_utterences
						snippet["range"] = snippet_range
						# end_index = [i for i,x in enumerate(snippet["range"]) if x == -1][1]
						snippet["mask"] = [1 if snippet["range"][j] >= 0 else 0 for j in range(self.conversation_size)]
						# snippet["mask"][end_index] = 0
						# snippet["utterances"][end_index] = end_utterance[0]
						snippet["length"] = len(snippet["utterances"])
						snippets.append(snippet)
						## TODO: put actual length here
						snippet_lengths.append(snippet["length"])
				id_snippet_dict[conversation.id] = list(range(snippet_bucket_length, snippet_bucket_length + len(snippets)))
				snippet_bucket += snippets
				snippet_bucket_length += len(snippets)

			# Create a small sample
			if args.truncate_dataset:
				if mode == "train":
					snippet_bucket = snippet_bucket[:5]
					snippet_lengths = snippet_lengths[:5]
				else:
					snippet_bucket = snippet_bucket[:2]
					snippet_lengths = snippet_lengths[:2]

			# sampling all snippets of all conversations at once for test time and only sampling a few during training time
			if mode == "train":
				# sampler = SingleSnippetSampler((id_snippet_dict, snippet_lengths), args.batch_size, shuffle=True)
				# introduces sampling errors
				sampler = SortedBatchSampler(snippet_lengths, args.batch_size, shuffle=True)
				batch_size = args.batch_size
			else:
				sampler = SortedBatchSampler(snippet_lengths, args.eval_batch_size, shuffle=True)
				batch_size = args.eval_batch_size

			batcher = SnippetBatcher(args, vocabulary)
			loader = torch.utils.data.DataLoader(
				snippet_bucket,
				batch_size=batch_size,
				sampler= sampler,
				num_workers=args.data_workers,
				collate_fn= batcher.batchify,
				pin_memory=False,
			)
			return loader
		if self.mode == "train":
			train_loader = create_snippets(dataset.train_dataset, dataset.vocabulary, model, mode = "train")
			valid_loader = create_snippets(dataset.valid_dataset, dataset.vocabulary, model, mode = "test")
			test_loader = create_snippets(dataset.test_dataset, dataset.vocabulary, model, mode = "test")
			return train_loader, valid_loader, test_loader
		elif self.mode == "test":
			valid_loader = create_snippets(dataset.valid_dataset, dataset.vocabulary, model, mode = "test")
			test_loader = create_snippets(dataset.test_dataset, dataset.vocabulary, model, mode = "test")
			return None, valid_loader, test_loader

class SnippetBatcher():
	def __init__(self, args, vocabulary):
		self.args = args
		self.vocabulary = vocabulary

	def batchify(self, batch_data):
		## Recieves conversations of roughly equal lengths, with <pad> token dummy utterances

		vocab_length = len(self.vocabulary.vocabulary)

		## For category Classification labels
		labels = []

		## Input
		conversation_mask = []
		conversation_ranges = []
		utterance_list = []  # elmo
		utterance_ids_list = []  # avg_elmo
		utterance_word_ids_list = []  # glove
		utterance_bow_list = []
		conversation_lengths = []
		conversation_ids = []

		batch = {}

		max_num_utterances = max([len(c["utterances"]) for c in batch_data])
		current_max_length = max([sum(batch_data[i]['mask']) for i in range(len(batch_data))])
		if current_max_length != max_num_utterances:
			# Trim excess dummy conversation in a batch
			new_batch = []
			for item in batch_data:
				new_item = {
					"utterances": item["utterances"][:current_max_length],
					"range": item["range"][:current_max_length],
					"mask": item["mask"][:current_max_length],
					"id": item["id"]
				}
				new_batch.append(new_item)
			batch_data = new_batch
			max_num_utterances = current_max_length
		max_utterance_length = max([max([len(u.ids) for u in c["utterances"]]) for c in batch_data])

		# optimize searches in vocabulary do that in the dataloader itself
		for c_idx, conversation in enumerate(batch_data):
			length = len(conversation["utterances"])
			utterance_ids_list += conversation["range"]
			for u_idx, u in enumerate(conversation["utterances"]):
				utterance_list.append(u.tokens)
				utterance_vocab_ids = u.ids
				utterance_word_ids_list.append(pad_seq(utterance_vocab_ids, max_utterance_length))
				utterance_bow_list.append(list(set(utterance_vocab_ids)))
				labels.append(u.label)
			utterance_ids_list += [-1] * (max_num_utterances - length)
			for i in range(max_num_utterances - length):
				utterance_list.append([self.vocabulary.pad_token])
				utterance_word_ids_list.append(pad_seq([], max_utterance_length))
				utterance_bow_list.append([0])
				labels.append(0)
			conversation_lengths.append(length)
			conversation_ids.append(conversation["id"])
			conversation_ranges.append(conversation["range"] + [-1] * (max_num_utterances - length))
			conversation_mask.append(conversation["mask"] + [0] * (max_num_utterances - length))

		batch['size'] = len(utterance_list)

		# Generate BOW Mask
		max_bows_batch = max(len(l) for l in utterance_bow_list)
		batch['utterance_bow_list'] = copy.deepcopy(utterance_bow_list)
		utterance_bow_list_padded = np.array([pad_seq(l, max_bows_batch) for l in utterance_bow_list])


		batch['utterance_list'] = utterance_list
		batch['utterance_word_ids'] = np.array(utterance_word_ids_list)
		batch['utterance_ids_list'] = np.array(utterance_ids_list)
		batch['input_mask'] = (1 * (batch['utterance_word_ids'] != 0))
		batch['label'] = labels
		batch['utterance_bow_mask'] = (1 * (utterance_bow_list_padded != 0))

		batch['conversation_lengths'] = conversation_lengths
		batch['conversation_ids'] = conversation_ids
		batch['conversation_mask'] = conversation_mask
		batch['max_num_utterances'] = max_num_utterances
		batch['max_utterance_length'] = max_utterance_length

		return batch

@RegisterLoader('conversation_length')
class ConversationDataloader(AbstractDataLoader):
	def __init__(self, args):
		self.args = args
		self.mode = args.run_mode
		self.batch_size = args.batch_size

	def get_dataloader(self, args, dataset, model):

		def create_conversations(dataset, vocabulary, model, mode = "train"):
			bucket = []
			bucket_lengths = []
			for data_item in dataset:
				## Data item change conversational ids to indices here itself.
				if hasattr(model.token_encoder, 'conversation_id2idx'):
					data_item.id = model.token_encoder.conversation_id2idx[data_item.id]
				for u_, u in enumerate(data_item.utterances):
					data_item.utterances[u_].ids = vocabulary.get_indices(u.tokens)
				bucket.append(data_item)
				bucket_lengths.append(len(data_item.utterances))
			if mode == "train":
				sampler = SortedBatchSampler(bucket_lengths, args.batch_size, shuffle=True)
				batch_size = args.batch_size
			else:
				sampler = SortedBatchSampler(bucket_lengths, args.eval_batch_size, shuffle=True)
				batch_size = args.eval_batch_size
			batcher = Batcher(args, vocabulary, model)
			loader = torch.utils.data.DataLoader(
				bucket,
				batch_size=batch_size,
				sampler=sampler,
				num_workers=args.data_workers,
				collate_fn=batcher.batchify,
				pin_memory=False,
			)
			return loader

		if self.mode == "train":
			train_loader = create_conversations(dataset.train_dataset, dataset.vocabulary, model, mode = "train")
			valid_loader = create_conversations(dataset.valid_dataset, dataset.vocabulary, model, mode = "test")
			test_loader = create_conversations(dataset.test_dataset, dataset.vocabulary, model, mode = "test")
			return train_loader, valid_loader, test_loader
		elif self.mode == "test":
			valid_loader = create_conversations(dataset.valid_dataset, dataset.vocabulary, model, mode = "test")
			test_loader = create_conversations(dataset.test_dataset, dataset.vocabulary, model, mode = "test")
			return None, valid_loader, test_loader

class Batcher():
	def __init__(self, args, vocabulary, model):
		self.args = args
		self.vocabulary = vocabulary

	def batchify(self, batch_data):
		utterance_bow_list = []

		## category classification labels
		labels = []

		conversation_mask = []
		utterance_list = []
		utterance_ids_list = []
		utterance_word_ids_list = []
		conversation_lengths = []
		conversation_ids = []

		batch = {}

		if self.args.truncate_dataset:
			for idx, conversation in enumerate(batch_data):
				random_short_length = random.sample(range(5, 10), 1)[0]
				batch_data[idx].utterances = conversation.utterances[:random_short_length]

		max_num_utterances = max([len(c.utterances) for c in batch_data])
		max_utterance_length = max([max([len(u.ids) for u in c.utterances]) for c in batch_data])
		for c_idx, conversation in enumerate(batch_data):
			length = len(conversation.utterances)
			for u_idx, u in enumerate(conversation.utterances):
				utterance_list.append(u.tokens)
				utterance_word_ids = u.ids
				utterance_word_ids_list.append(pad_seq(utterance_word_ids, max_utterance_length))
				utterance_ids_list.append(u_idx)
				labels.append(u.label)
				utterance_bow_list.append(list(set(utterance_word_ids)))

			conversation_lengths.append(length)
			conversation_ids.append(conversation.id)
			conversation_mask.append([1] * length + [0] * (max_num_utterances - length))
			## apend dummy utterances
			for i in range(max_num_utterances - length):
				utterance_list.append([])
				utterance_word_ids_list.append(pad_seq([], max_utterance_length))
				utterance_ids_list.append(-1)
				labels.append(0)
				utterance_bow_list.append([0])

		batch['size'] = len(utterance_list)
		# Generate BOW Mask
		max_bows_batch = max(len(l) for l in utterance_bow_list)
		batch['utterance_bow_list'] = copy.deepcopy(utterance_bow_list)
		utterance_bow_list_padded = np.array([pad_seq(l, max_bows_batch) for l in utterance_bow_list])

		batch['utterance_list'] = utterance_list
		batch['utterance_word_ids'] = np.array(utterance_word_ids_list)
		batch['utterance_ids_list'] = np.array(utterance_ids_list)
		batch['input_mask'] = (1 * (batch['utterance_word_ids'] != 0))

		batch['label'] = labels
		batch['utterance_bow_mask'] = (1 * (utterance_bow_list_padded != 0))

		batch['conversation_lengths'] = conversation_lengths
		batch['conversation_ids'] = conversation_ids
		batch['conversation_mask'] = conversation_mask
		batch['max_num_utterances'] = max_num_utterances
		batch['max_utterance_length'] = max_utterance_length

		return batch

@RegisterBatcher('conversation_snippets')
class ConversationSnippetBatcher(AbstractDataLoader):
	def __init__(self, args):
		self.args = args

	def get_batches(self, args, dataset):
		# shuffle dataset
		# fix max length and generate snippets of conversation from that dataset
		mode = args.run_mode
		batch_size = args.batch_size
		conversation_size = args.conversation_size

		def create_conversation_batch(batch_data, vocabulary):
			## Recieves conversations of roughly equal lengths, with <pad> token dummy utterances

			vocab_length = len(vocabulary.vocabulary)

			## For category Classification labels
			labels = []

			## Input
			conversation_mask = []
			conversation_ranges = []
			utterance_list = [] #elmo
			utterance_ids_list = [] # avg_elmo
			utterance_word_ids_list = [] #glove
			utterance_bow_list = []
			conversation_lengths = []
			conversation_ids = []

			batch = {}


			max_num_utterances = max([len(c["utterances"]) for c in batch_data])
			current_max_length = max([sum(batch_data[i]['mask']) for i in range(len(batch_data))])
			if current_max_length != max_num_utterances:
				# Trim excess dummy conversation in a batch
				new_batch = []
				for item in batch_data:
					new_item = {
						"utterances" : item["utterances"][:current_max_length],
						"range" : item["range"][:current_max_length],
						"mask" : item["mask"][:current_max_length],
						"id" : item["id"]
					}
					new_batch.append(new_item)
				batch_data = new_batch
				max_num_utterances = current_max_length
			max_utterance_length = max([max([len(u.tokens) for u in c["utterances"]]) for c in batch_data])

			for c_idx, conversation in enumerate(batch_data):
				length = len(conversation["utterances"])
				utterance_ids_list += conversation["range"]
				for u_idx, u in enumerate(conversation["utterances"]):
					utterance_list.append(u.tokens)
					utterance_vocab_ids = vocabulary.get_indices(u.tokens)
					utterance_word_ids_list.append(pad_seq(utterance_vocab_ids, max_utterance_length))
					utterance_bow_list.append(list(set(utterance_vocab_ids)))
					labels.append(u.label)
				utterance_ids_list += [-1]*(max_num_utterances - length)
				for i in range(max_num_utterances - length):
					utterance_list.append([vocabulary.pad_token])
					utterance_word_ids_list.append(pad_seq([], max_utterance_length))
					utterance_bow_list.append([vocabulary.vocabulary[vocabulary.pad_token]])
					labels.append(0)
				conversation_lengths.append(length)
				conversation_ids.append(conversation["id"])
				conversation_ranges.append(conversation["range"] + [-1]*(max_num_utterances - length))
				conversation_mask.append(conversation["mask"] + [0]*(max_num_utterances - length))

			# Generate BOW Mask
			max_bows_batch = max(len(l) for l in utterance_bow_list)
			utterance_bow_list = [pad_seq(l, max_bows_batch) for l in utterance_bow_list]

			batch['utterance_list'] = utterance_list
			batch['utterance_word_ids'] = np.array(utterance_word_ids_list)
			batch['utterance_ids_list'] = np.array(utterance_ids_list)
			batch['utterance_bow_list'] = np.array(utterance_bow_list)
			batch['input_mask'] = (1 * (batch['utterance_word_ids'] != 0))
			batch['label'] = labels
			batch['utterance_bow_mask'] = (1 * (batch['utterance_bow_list'] != 0))

			batch['conversation_lengths'] = conversation_lengths
			batch['conversation_ids'] = conversation_ids
			batch['conversation_mask'] = conversation_mask
			batch['max_num_utterances'] = max_num_utterances
			batch['max_utterance_length'] = max_utterance_length

			return batch



		def create_batches(dataset, vocabulary):
			batches = []
			buckets = defaultdict(list)
			bucket = []
			for c_idx, conversation in enumerate(dataset):
				length = len(conversation.utterances)
				snippets = []
				dummy_utterance = [AbstractDataset.Utterance([vocabulary.pad_token])]
				start_utterance = [AbstractDataset.Utterance([vocabulary.soc])]
				end_utterance = [AbstractDataset.Utterance([vocabulary.eoc])]
				if length <= conversation_size:
					snippet = {}
					snippet["id"] = conversation.id
					snippet["utterances"] = start_utterance + conversation.utterances + end_utterance
					snippet["range"] = [-1] +list(range (0, length)) + [-1]
					snippet["mask"] = [1]*(length+2)
					snippets.append(snippet)
				else:
					# Pad with dummpy utterances on both sides
					padded_dummy_utterances = dummy_utterance*(conversation_size-1)
					padded_utterances = padded_dummy_utterances + conversation.utterances + padded_dummy_utterances
					padded_range = [-1]*(conversation_size-1) + list(range(length)) + [-1]*(conversation_size-1)
					for i in range(length + conversation_size - 1):
						snippet = {}
						snippet["id"] = conversation.id
						snippet_utterences = padded_utterances[i : i+conversation_size]
						snippet_range = padded_range[i : i+conversation_size]
						if i < conversation_size - 1:
							snippet_utterences = snippet_utterences[conversation_size-i-1:] + snippet_utterences[0:conversation_size-i-1]
							snippet_range = snippet_range[conversation_size-i-1:] + snippet_range[0:conversation_size-i-1]
						snippet["utterances"] = start_utterance + snippet_utterences + dummy_utterance
						snippet["range"] = [-1] + snippet_range + [-1]
						end_index = [i for i,x in enumerate(snippet["range"]) if x == -1][1]
						snippet["mask"] = [1] + [1 if snippet["range"][j] >= 0 else 0 for j in range(1,conversation_size+1)] + [0]
						snippet["mask"][end_index] = 1
						snippet["utterances"][end_index] = end_utterance[0]
						snippets.append(snippet)
				bucket += snippets

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
			train_batches = create_batches(dataset.train_dataset, dataset.vocabulary)
			valid_batches = create_batches(dataset.valid_dataset, dataset.vocabulary)
			test_batches = create_batches(dataset.test_dataset, dataset.vocabulary)
			np.random.shuffle(train_batches)
			return train_batches, valid_batches, test_batches
		elif mode =="test":
			valid_batches = create_batches(dataset.valid_dataset, dataset.vocabulary)
			test_batches = create_batches(dataset.test_dataset, dataset.vocabulary)
			return None,valid_batches,test_batches


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

			vocab_length = len(vocabulary.vocabulary)

			## for classification
			utterance_options_list = []
			next_gold_ids = []
			prev_gold_ids = []

			## utterance vocabulary ids for next/previous utterance reconstruction (bow model and decoding)
			next_utterance_ids_list = []
			previous_utterance_ids_list = []
			next_utterance_bow_list = []
			prev_utterance_bow_list= []

			utterance_bow_list = []

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
					next_utterance_ids_list.append(pad_seq(next_utterance_ids, max_utterance_length))
					previous_utterance_ids_list.append(pad_seq(previous_utterance_ids, max_utterance_length))
					# Next utterance bow will be a vector over Vocabulary
					next_utterance_bow_list.append(next_utterance_bow)
					prev_utterance_bow_list.append(prev_utterance_bow)

					utterance_bow_list.append(list(set(vocabulary.get_indices(u.tokens))))


				conversation_lengths.append(length)
				conversation_ids.append(conversation.id)
				conversation_mask.append([1]*length + [0]*(max_num_utterances - length))
				## apend dummy utterances
				for i in range(max_num_utterances - length):
					utterance_list.append([])
					utterance_word_ids_list.append(pad_seq([], max_utterance_length))
					next_utterance_ids_list.append(pad_seq([], max_utterance_length))
					previous_utterance_ids_list.append(pad_seq([], max_utterance_length))
					utterance_ids_list.append(-1)
					utterance_options_list.append([length + i + c_idx*max_num_utterances]*self.K)
					next_gold_ids.append(0)
					prev_gold_ids.append(0)
					labels.append(0)
					next_utterance_bow_list.append([])
					prev_utterance_bow_list.append([])
					utterance_bow_list.append([vocabulary.vocabulary[vocabulary.pad_token]])

			batch['size'] = len(utterance_list)
			# Generate BOW Mask
			max_next_bows_batch = max(len(l) for l in next_utterance_bow_list)
			max_prev_bows_batch = max(len(l) for l in previous_utterance_ids_list)
			next_utterance_bow_list = [pad_seq(l, max_next_bows_batch) for l in next_utterance_bow_list]
			prev_utterance_bow_list = [pad_seq(l, max_prev_bows_batch) for l in prev_utterance_bow_list]

			max_bows_batch = max(len(l) for l in utterance_bow_list)
			batch['utterance_bow_list'] = copy.deepcopy(utterance_bow_list)
			utterance_bow_list_padded = np.array([pad_seq(l, max_bows_batch) for l in utterance_bow_list])


			batch['utterance_list'] = utterance_list
			batch['utterance_word_ids'] = np.array(utterance_word_ids_list)
			batch['utterance_ids_list'] = np.array(utterance_ids_list)
			batch['input_mask'] = (1 * (batch['utterance_word_ids'] != 0))

			batch['utterance_options_list'] = utterance_options_list
			batch['next_utterance_gold'] = next_gold_ids
			batch['next_utterance_ids'] = np.array(next_utterance_ids_list)
			batch['next_utterance_mask'] = (1 * (batch['next_utterance_ids'] != 0))
			batch['prev_utterance_gold'] = prev_gold_ids
			batch['prev_utterance_ids'] = np.array(previous_utterance_ids_list)
			batch['prev_utterance_mask'] = (1 * (batch['prev_utterance_ids'] != 0))

			batch['label'] = labels
			batch['next_bow_list'] = np.array(next_utterance_bow_list)
			batch['prev_bow_list'] = np.array(prev_utterance_bow_list)
			batch['next_bow_mask'] = (1 * (batch['next_bow_list'] != 0))
			batch['prev_bow_mask'] = (1 * (batch['prev_bow_list'] != 0))
			batch['utterance_bow_mask'] = (1 * (utterance_bow_list_padded != 0))

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

