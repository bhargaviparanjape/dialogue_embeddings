import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
from allennlp.modules.elmo import Elmo, batch_to_ids
import pdb
import codecs


from src.models.factory import RegisterModel
from src.utils.utility_functions import variable, FloatTensor, ByteTensor, LongTensor


@RegisterModel('elmo')
class ELMoEmbedding():
	def __init__(self, args, **kwargs):
		options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
		weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
		self.args = args
		self.ee = Elmo(options_file, weight_file, requires_grad=False, num_output_representations=1, dropout=args.dropout)

	def lookup(self, input):
		embed_lookup = input["utterance_list"]
		character_ids = batch_to_ids(embed_lookup)

		## Batch lookup for memory out of bound error
		K = 10
		embeddings = torch.zeros(character_ids.shape[0], character_ids.shape[1], 1024)
		mask = torch.zeros(character_ids.shape[0], character_ids.shape[1])
		for i in range(0, character_ids.shape[0], K):
			dict = self.ee(character_ids[i:i + K].unsqueeze(0))
			embeddings[i:i + K] = dict['elmo_representations'][0]
			mask[i:i + K] = dict['mask']
		return embeddings, mask

	def lookup_by_name(self, input, name_embed, name_mask):
		raise NotImplementedError

	@staticmethod
	def add_args(parser):
		embedding_layer_parameters = parser.add_argument_group("Embedding layer Parameters")
		embedding_layer_parameters.add_argument("--embed-size", type=int, default=1024)


@RegisterModel('avg_elmo')
class AverageELMoEmbedding():
	def __init__(self, args, **kwargs):
		self.pretrained_embedding_path = args.pretrained_embedding_path
		self.embed_size = args.embed_size
		self.args = args
		self.load_embeddings()

	def load_embeddings(self, vocabulary):
		self.embeddings = {}
		with open(self.pretrained_embedding_path) as fin:
			for line in fin:
				dict =  json.loads(line)
				self.embeddings[dict["id"]] = dict["embeddings"]


	def lookup(self, input):
		conversation_id_list = input["conversation_ids"]
		input_mask = FloatTensor(input["input_mask"])
		max_num_utterances_batch = input['max_num_utterances']

		batch_embeddings = []
		for x, id in enumerate(conversation_id_list):
			embeddings = self.embeddings[id]
			batch_embeddings += embeddings
			batch_embeddings += [np.random.rand(self.args.embed_size).tolist() for i in range(max_num_utterances_batch - len(embeddings))]
		batch_embedding_tensor = FloatTensor(batch_embeddings)
		return batch_embedding_tensor, input_mask

	def lookup_by_name(self, input, name_embed, name_mask):
		conversation_id_list = input[name_embed]
		input_mask = FloatTensor(input[name_mask])
		# Generally remains the same
		max_num_utterances_batch = input['max_num_utterances']

		batch_embeddings = []
		for x, id in enumerate(conversation_id_list):
			embeddings = self.embeddings[id]
			batch_embeddings += embeddings
			batch_embeddings += [np.random.rand(self.args.embed_size).tolist() for i in
								 range(max_num_utterances_batch - len(embeddings))]
		batch_embedding_tensor = FloatTensor(batch_embeddings)
		return batch_embedding_tensor, input_mask


@RegisterModel('glove')
class GloveEmbeddings():
	def __init__(self, args, **kwargs):
		self.pretrained_embedding_path = args.pretrained_embedding_path
		self.embed_size = args.embed_size
		self.args = args


	def load_embeddings(self, vocabulary):
		word_to_id = vocabulary
		self.vocabulary = vocabulary
		self.embeddings = []
		print("Loading pretrained embeddings from {0}".format(self.pretrained_embedding_path))
		for _ in range(len(word_to_id)):
			self.embeddings.append(np.random.uniform(-math.sqrt(3.0 / self.embed_size),
													 math.sqrt(3.0 / self.embed_size), size=self.embed_size))

		print("length of dict: {0}".format(len(word_to_id)))
		pretrain_word_emb = {}
		if self.pretrained_embedding_path is not None:
			for line in codecs.open(self.embedding_path, "r", "utf-8", errors='replace'):
				items = line.strip().split()
				if len(items) == self.embed_size + 1:
					try:
						pretrain_word_emb[items[0]] = np.asarray(items[1:]).astype(np.float32)
					except ValueError:
						continue
		# fout_required = open("glove.840B.300d.required.txt")

		not_covered = 0
		for word, id in word_to_id.items():
			word = str(word)
			if word in pretrain_word_emb.keys():
				self.embeddings[id] = pretrain_word_emb[word]
			# fout_required.write(word + " " + " ".join([str(f) for f in pretrain_word_emb[word]]) + "\n")
			elif word.lower() in pretrain_word_emb.keys():
				self.embeddings[id] = pretrain_word_emb[word.lower()]
			#           fout_required.write(word.lower() + " " + " ".join([str(f) for f in pretrain_word_emb[word]]) + "\n")
			else:
				not_covered += 1

		embs = np.array(self.embeddings, dtype=np.float32)
		print("Word number not covered in pretrain embedding: {0}".format(not_covered))

		## Initialize look-up encoder
		self.embed_layer = LookupEncoder(self.args, len(self.vocabulary), self.args.embed_size, embs)
		if self.args.use_cuda:
			self.embed_layer = self.embed_layer.cuda()

	def lookup(self, input):
		input_token_ids = LongTensor(input["utterance_word_ids"])
		utterance_embeddings = self.embed_layer(input_token_ids)
		input_mask = FloatTensor(input['input_mask'])
		return utterance_embeddings, input_mask


	def lookup_by_name(self, input, name_embed, name_mask):
		input_token_ids = LongTensor(input[name_embed])
		utterance_embeddings = self.embed_layer(input_token_ids)
		input_mask = FloatTensor(input[name_mask])
		return utterance_embeddings, input_mask


class LookupEncoder(nn.Module):
	def __init__(self, args, vocab_size, embedding_dim, pretrain_embedding=None):
		super(LookupEncoder, self).__init__()
		self.embedding_dim = embedding_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		if pretrain_embedding is not None:
			self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))

		self.word_embeddings.weight.requires_grad = False

	def forward(self, batch):
		return self.word_embeddings(batch)