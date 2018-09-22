import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from embed.models.factory import RegisterModel
from allennlp.modules.elmo import Elmo, batch_to_ids
import pdb

@RegisterModel('elmo')
class ELMoEmbedding():
	def __init__(self, args):
		#super(ELMoEmbedding, self).__init__()
		options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
		weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
		self.args = args
		self.ee = Elmo(options_file, weight_file, requires_grad=False, num_output_representations = 1, dropout=args.dropout)


	def get_embeddings(self, *input):
		## input : list of list of strings
		character_ids = batch_to_ids(input[0])
		#if self.args.use_cuda:
		#	character_ids = character_ids.cuda()
		#for u in character_ids.data.numpy():
		embeddings = torch.FloatTensor(character_ids.shape[0], character_ids.shape[1], 1024)
		mask= torch.Tensor(character_ids.shape[0], character_ids.shape[1])
		for i in range(character_ids.shape[0]):
			dict = self.ee(character_ids[i].unsqueeze(0))
			embeddings[i] = dict['elmo_representations'][0]
			mask[i] = dict['mask']

		return embeddings, mask


