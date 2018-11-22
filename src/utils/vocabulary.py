from collections import defaultdict
from src.utils.global_parameters import MAX_VOCAB_LENGTH
import math
import collections
import numpy as np

class Vocabulary(object):
	def __init__(self, pad_token='<pad>', unk='<unk>', sos='<sos>', eos='<eos>', soc='<soc>', eoc='<eoc>'):

		self.vocabulary = dict()
		self.id_to_vocab = dict()
		self.utterance_frequency = dict()
		self.pad_token = pad_token
		self.unk = unk
		self.sos = sos
		self.eos = eos
		self.soc = soc
		self.eoc = eoc
		self.std_tokens = [self.sos, self.eos, self.pad_token, self.unk, self.soc, self.eoc]

		self.vocabulary[pad_token] = 0
		self.vocabulary[unk] = 1
		self.vocabulary[sos] = 2
		self.vocabulary[eos] = 3
		self.vocabulary[soc] = 4
		self.vocabulary[eoc] = 5

		self.id_to_vocab[0] = pad_token
		self.id_to_vocab[1] = unk
		self.id_to_vocab[2] = sos
		self.id_to_vocab[3] = eos
		self.id_to_vocab[4] = soc
		self.id_to_vocab[5] = eoc

		self.nertag_to_id = dict()
		self.postag_to_id = dict()
		self.char_to_id = dict()
		self.id_to_char = dict()
		self.id_to_nertag = dict()
		self.id_to_postag = dict()

		self.counter = dict()

	def add_and_get_index(self, word):
		if word in self.vocabulary:
			self.counter[word] += 1
			return self.vocabulary[word]
		else:
			length = len(self.vocabulary)
			self.vocabulary[word] = length
			self.id_to_vocab[length] = word
			self.counter[word] = 1
			return length

	def add_and_get_indices(self, words):
		indices = [self.add_and_get_index(word) for word in words]
		# These words are over a single utterance;
		# For each unique word; increas utterance count
		for word in set(words):
			index = self.get_index(word)
			if index in self.utterance_frequency:
				self.utterance_frequency[index] += 1
			else:
				self.utterance_frequency[index] = 1
		return indices

	def get_index(self, word):
		return self.vocabulary.get(word, self.vocabulary[self.unk])

	def get_indices(self, words):
		return [self.get_index(word) for word in words]

	def get_token(self, id):
		# No protection against KeyError, code has to fail if key not found
		return self.id_to_vocab[id]

	def get_tokens(self, indices):
		return [self.get_token(id) for id in indices]

	def get_length(self):
		return len(self.vocabulary)

	def get_character_vocab(self):
		unique_characters = set()
		for key, _ in self.vocabulary.items():
			unique_characters = unique_characters | set(key)
		for i,c in enumerate(unique_characters):
			self.char_to_id[c] = i
			self.id_to_char[i] = c

	def get_char_index(self, character):
		return self.char_to_id[character]

	def get_char_indices(self, word):
		return [self.char_to_id[word[i]] for i in range(len(word))]

	def truncate(self):
		vocabulary = sorted(self.counter.items(), key=lambda k_v: k_v[1], reverse=True)
		self.vocabulary = {}
		self.id_to_vocab = {}
		self.vocabulary[self.pad_token] = 0
		self.vocabulary[self.unk] = 1
		self.vocabulary[self.sos] = 2
		self.vocabulary[self.eos] = 3
		self.vocabulary[self.soc] = 4
		self.vocabulary[self.eoc] = 5


		self.id_to_vocab[0] = self.pad_token
		self.id_to_vocab[1] = self.unk
		self.id_to_vocab[2] = self.sos
		self.id_to_vocab[3] = self.eos
		self.id_to_vocab[4] = self.soc
		self.id_to_vocab[5] = self.eoc
		# Maintain uniform order of indices
		for id, item in enumerate(vocabulary[:MAX_VOCAB_LENGTH]):
			if item[0] in self.std_tokens:
				continue
			self.vocabulary[item[0]] = id + 6
			self.id_to_vocab[id + 6] = item[0]
		self.counter = {k:self.counter[k] for k,v in vocabulary[:MAX_VOCAB_LENGTH]}
		self.utterance_frequency = {self.vocabulary[k]: self.utterance_frequency[self.vocabulary[k]] for k, v in vocabulary[:MAX_VOCAB_LENGTH]}


	def get_word(self, index):
		if index < len(self.id_to_vocab):
			return self.id_to_vocab[index]
		else:
			return ""

	def ner_tag_size(self):
		return len(self.nertag_to_id)

	def pos_tag_size(self):
		return len(self.postag_to_id)

	def compute_inverse_frequency(self, total_num_utterances):
		self.inverse_utterance_frequency = {}
		max_num_utterances = max(self.utterance_frequency.values())
		for key, value in self.utterance_frequency.items():
			self.inverse_utterance_frequency[key] = 1/(1+ math.log(value +  1))
			# self.inverse_utterance_frequency[key] = math.log(total_num_utterances/value)
		for token in self.std_tokens:
			## All standard tokens have the maximum frequency
			self.inverse_utterance_frequency[self.vocabulary[token]] = 1/(1+ math.log(max_num_utterances + 1))
			# self.inverse_utterance_frequency[self.vocabulary[token]] = math.log(total_num_utterances/max_num_utterances)
		## All IDS are intact and in order, convert dictionary to list for easy indexing
		self.inverse_utterance_frequency = np.array([v for k,v in sorted(self.inverse_utterance_frequency.items())])


	def __add__(self, other):
		## keys in both vocabularies
		## TODO: NO support for limiting the vocabulary of multiple datasets using counter

		unique_words = list(set(list(self.vocabulary.keys()) + list(other.vocabulary.keys())) - \
					   set(self.std_tokens))
		unique_characters = list(set(list(self.char_to_id.keys()) + list(other.char_to_id.keys())))
		unique_nertag = list(set(list(self.nertag_to_id.keys()) + list(other.nertag_to_id.keys())))
		unique_postag = list(set(list(self.postag_to_id.keys()) + list(other.postag_to_id.keys())))

		aggregated_counter = dict()
		aggregated_utterance_frequency = dict()
		aggregated_vocab = Vocabulary()
		aggregated_vocab.counter = dict()
		aggregated_vocab.vocabulary = dict()
		aggregated_vocab.id_to_vocab = dict()
		aggregated_vocab.id_to_char = dict()
		aggregated_vocab.char_to_id = dict()
		aggregated_vocab.nertag_to_id = dict()
		aggregated_vocab.id_to_nertag = dict()
		aggregated_vocab.postag_to_id = dict()
		aggregated_vocab.id_to_postag = dict()
		aggregated_vocab.vocabulary[aggregated_vocab.pad_token] = 0
		aggregated_vocab.vocabulary[aggregated_vocab.unk] = 1
		aggregated_vocab.vocabulary[aggregated_vocab.sos] = 2
		aggregated_vocab.vocabulary[aggregated_vocab.eos] = 3
		aggregated_vocab.vocabulary[aggregated_vocab.soc] = 4
		aggregated_vocab.vocabulary[aggregated_vocab.eoc] = 5

		aggregated_vocab.id_to_vocab[0] = aggregated_vocab.pad_token
		aggregated_vocab.id_to_vocab[1] = aggregated_vocab.unk
		aggregated_vocab.id_to_vocab[2] = aggregated_vocab.sos
		aggregated_vocab.id_to_vocab[3] = aggregated_vocab.eos
		aggregated_vocab.id_to_vocab[4] = aggregated_vocab.soc
		aggregated_vocab.id_to_vocab[5] = aggregated_vocab.eoc

		for id in range(len(unique_words)):
			aggregated_vocab.vocabulary[unique_words[id]] = id + 6
			aggregated_vocab.id_to_vocab[id + 6] = unique_words[id]

		for id in range(len(unique_characters)):
			aggregated_vocab.char_to_id[unique_characters[id]] = id
			aggregated_vocab.id_to_char[id] = unique_characters[id]

		for id in range(len(unique_nertag)):
			aggregated_vocab.nertag_to_id[unique_nertag[id]] = id
			aggregated_vocab.id_to_nertag[id] = unique_nertag[id]

		for id in range(len(unique_postag)):
			aggregated_vocab.postag_to_id[unique_postag[id]] = id
			aggregated_vocab.id_to_postag[id] = unique_postag[id]

		## for common words, add counters, otherwise retain infividual counters
		for enum_, id in enumerate(unique_words):
			if id in self.std_tokens:
				continue
			if id in self.counter.keys() and id in other.counter.keys():
				aggregated_counter[id] = self.counter[id] + other.counter[id]
				idx1 = self.vocabulary[id]
				idx2 = other.vocabulary[id]
				aggregated_utterance_frequency[enum_ + 6] = self.utterance_frequency[idx1] + other.utterance_frequency[idx2]
			elif id in self.counter.keys():
				aggregated_counter[id] = self.counter[id]
				idx = self.vocabulary[id]
				aggregated_utterance_frequency[enum_ + 6] = self.utterance_frequency[idx]
			else:
				aggregated_counter[id] = other.counter[id]
				idx = other.vocabulary[id]
				aggregated_utterance_frequency[enum_ + 6] = other.utterance_frequency[idx]
		aggregated_vocab.counter = aggregated_counter
		aggregated_vocab.utterance_frequency = aggregated_utterance_frequency
		return aggregated_vocab
