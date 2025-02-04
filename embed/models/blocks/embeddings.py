import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor
import json
from embed.models.factory import RegisterModel
from allennlp.modules.elmo import Elmo, batch_to_ids
import pdb
import codecs

@RegisterModel('elmo')
class ELMoEmbedding():
	def __init__(self, args):
		#super(ELMoEmbedding, self).__init__()
		options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
		weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
		self.args = args
		self.ee = Elmo(options_file, weight_file, requires_grad=False, num_output_representations = 1, dropout=args.dropout)


	def get_embeddings(self, *input):
		## input : list of list of i
		character_ids = batch_to_ids(input[0])
		#if self.args.use_cuda:
		#	character_ids = character_ids.cuda()
		#for u in character_ids.data.numpy():
		embeddings = torch.zeros(character_ids.shape[0], character_ids.shape[1], 1024)
		mask= torch.zeros(character_ids.shape[0], character_ids.shape[1])
		for i in range(0, character_ids.shape[0], 10):
			dict = self.ee(character_ids[i:i+10].unsqueeze(0))
			embeddings[i:i+10] = dict['elmo_representations'][0]
			mask[i:i+10] = dict['mask']

		return embeddings, mask

@RegisterModel('avg_elmo')
class AverageELMoEmbedding():
	def __init__(self, args):
		self.embedding_path = args.embedding_path
		self.embed_size = args.embed_size
		self.args = args
		self.load_embeddings()

	def load_embeddings(self):
		self.embeddings = {}
		with open(self.embedding_path) as fin:
			for line in fin:
				dict =  json.loads(line)
				self.embeddings[dict["id"]] = dict["embeddings"]


	def lookup(self, transcript_id_list, max_batch_length):
		batch_embeddings = []
		for x, id in enumerate(transcript_id_list):
			embeddings = self.embeddings[id]
			batch_embeddings += embeddings
			batch_embeddings += [np.random.rand(self.args.embed_size).tolist() for i in range(max_batch_length - len(embeddings))]
		batch_embedding_tensor = FloatTensor(batch_embeddings)
		return batch_embedding_tensor



@RegisterModel('glove')
class GloveEmbeddings():
	def __init__(self, args):
		self.embedding_path = args.embedding_path
		self.embed_size = args.embed_size
		self.args = args
		embs = self.load_embeddings(args.vocabulary)
		self.lookup = LookupEncoder(args, len(self.args.vocabulary), self.args.embed_size, embs)
		if args.use_cuda:
			self.lookup = self.lookup.cuda()


	def load_embeddings(self, vocabulary):
		word_to_id = vocabulary
		self.embeddings = []
		print("Loading pretrained embeddings from {0}".format(self.embedding_path))
		for _ in range(len(word_to_id)):
			self.embeddings.append(np.random.uniform(-math.sqrt(3.0 / self.embed_size),
													 math.sqrt(3.0 / self.embed_size), size=self.embed_size))

		print("length of dict: {0}".format(len(word_to_id)))
		pretrain_word_emb = {}
		if self.embedding_path is not None:
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

		emb = np.array(self.embeddings, dtype=np.float32)
		print("Word number not covered in pretrain embedding: {0}".format(not_covered))
		## required to initialize lookup encoder
		return emb

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




