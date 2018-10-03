import torch
import torch.nn as nn
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor
from embed.models import factory as model_factory
import numpy as np
from torch.nn import functional
import pdb


@RegisterModel('dialogue_embedder')
class DialogueEmbedder(nn.Module):
	def __init__(self, args):
		super(DialogueEmbedder, self).__init__()
		self.args = args

		self.lookup_layer = model_factory.get_model(args, args.lookup)
		self.encoding_layer = model_factory.get_model(args, args.encoding)

	def forward(self, *input):
		[embeddings, input_mask_variable, \
		 conversation_mask, max_num_utterances_batch] = input[0]

		if self.args.embedding != "avg_elmo":
			lookup = self.lookup_layer(embeddings, input_mask_variable)
		else:
			lookup = embeddings
		## reshape based on max_input_length in batch dimension
		sequence_batch_size = int(lookup.shape[0] / max_num_utterances_batch)
		reshaped_lookup = lookup.view(sequence_batch_size, max_num_utterances_batch, lookup.shape[1])
		## sort utterances then apply encoding layer
		# sorted_lookup = reshaped_lookup[sort]
		encoded, _ = self.encoding_layer(reshaped_lookup, conversation_mask)
		## get hidden representations
		# if self.args.encoding == "bilstm":
		# 	encoded, _ = self.encoding_layer(sorted_lookup, lengths_sorted)
		# else:
		# 	encoded = self.encoding_layer(sorted_lookup, conversation_mask_sorted)
		# 	encoded = torch.cat((encoded[0, :], encoded[1, :]), 2)

		## convert the hidden representation into batched format again after going over all the conversations in the batch
		# encoded = encoded[unsort].view(sequence_batch_size * max_num_utterances_batch, -1, encoded.shape[2])
		encoded = encoded.view(sequence_batch_size * max_num_utterances_batch, -1, encoded.shape[2])
		return encoded

