import torch
import torch.nn as nn
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor
from embed.models import factory as model_factory
import numpy as np
from torch.nn import functional

@RegisterModel('vanilla')
class VanillaModel(nn.Module):
	def __init__(self, args):
		super(VanillaModel, self).__init__()
		self.args = args
		self.embedding_layer = model_factory.get_model(args, args.embedding)
		self.lookup_layer = model_factory.get_model(args, args.lookup)
		self.encoding_layer = model_factory.get_model(args, args.encoding)

		self.scorer = nn.Bilinear(2*args.encoder_hidden_size, 2*args.encoder_hidden_size, 1)
		self.objective = nn.CrossEntropyLoss()


	def forward(self, *input):
		[embeddings, input_mask_variable, \
		sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch, \
		options_tensor, goldids_variable] = input[0]
		lookup = self.lookup_layer(embeddings, input_mask_variable)
		## reshape based on max_input_length in batch dimension
		sequence_batch_size = int(lookup.shape[0]/max_num_utterances_batch)
		reshaped_lookup = lookup.view(sequence_batch_size, max_num_utterances_batch, lookup.shape[1])
		## sort utterances then apply encoding layer
		sorted_lookup = reshaped_lookup[sort]
		## get hidden representations
		encoded,_ = self.encoding_layer(sorted_lookup, lengths_sorted)
		encoded = encoded[unsort].view(sequence_batch_size * max_num_utterances_batch, -1, encoded.shape[2])
		## do lookup based on indices
		options = torch.index_select(encoded.squeeze(1), 0, options_tensor.view(-1))
		## expand the options K times get
		encoded = encoded.expand(encoded.shape[0], options_tensor.shape[1], encoded.shape[2]).contiguous()
		encoded_expand = encoded.view(encoded.shape[0]*options_tensor.shape[1],encoded.shape[2])
		## MLP
		logits = self.scorer(encoded_expand, options)
		logits_flat = logits.view(encoded.shape[0], options_tensor.shape[1], -1)
		log_probs_flat = functional.log_softmax(logits_flat)
		conversation_mask = conversation_mask_sorted[unsort]
		losses_flat = -torch.gather(log_probs_flat.squeeze(2), dim=1, index=goldids_variable.view(-1,1))
		losses = losses_flat * conversation_mask.float().unsqueeze(1)
		## loss and indices
		loss = losses.sum() / conversation_mask.float().sum()
		predictions = torch.sort(logits_flat.squeeze(2), descending=True)[1][:,0]
		return loss, predictions

		

	def eval(self, *input):
		[embeddings, input_mask_variable, \
		 sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch, \
		 options_tensor, goldids_variable] = input[0]
		lookup = self.lookup_layer(embeddings, input_mask_variable)
		## reshape based on max_input_length in batch dimension
		sequence_batch_size = int(lookup.shape[0] / max_num_utterances_batch)
		reshaped_lookup = lookup.view(sequence_batch_size, max_num_utterances_batch, lookup.shape[1])
		## sort utterances then apply encoding layer
		sorted_lookup = reshaped_lookup[sort]
		## get hidden representations
		encoded, _ = self.encoding_layer(sorted_lookup, lengths_sorted)
		encoded = encoded[unsort].view(sequence_batch_size * max_num_utterances_batch, -1, encoded.shape[2])
		## do lookup based on indices
		options = torch.index_select(encoded.squeeze(1), 0, options_tensor.view(-1))
		## expand the options K times get
		encoded = encoded.expand(encoded.shape[0], options_tensor.shape[1], encoded.shape[2]).contiguous()
		encoded_expand = encoded.view(encoded.shape[0] * options_tensor.shape[1], encoded.shape[2])
		## MLP
		logits = self.scorer(encoded_expand, options)
		logits_flat = logits.view(encoded.shape[0], options_tensor.shape[1], -1)
		predictions = torch.sort(logits_flat.squeeze(2), descending=True)[1][:, 0]
		return predictions

	def prepare_for_gpu(self, batch):
		## outputform for the prepare for gpu function
		## batch_size, gold_output, *input
		batch_size = int(len(batch['utterance_list'])/batch['max_num_utterances'])
		elmo_embeddings, input_mask = self.embedding_layer.get_embeddings(batch['utterance_list'])

		input_mask_variable = variable(input_mask).float()
		batch_lengths = batch['conversation_lengths']

		length_sort = np.argsort(batch_lengths)[::-1].copy()
		unsort = variable(LongTensor(np.argsort(length_sort)))
		conversation_mask_sorted = variable(FloatTensor(batch['conversation_mask'])[length_sort])
		conversation_mask = LongTensor(batch['conversation_mask'])
		lengths_sorted = np.array(batch_lengths)[length_sort]
		sort = length_sort


		options_tensor = LongTensor(batch['next_utterance_options_list'])
		gold_ids = np.array(batch['next_utterance_gold_ids'])
		goldids_variable = LongTensor(batch['next_utterance_gold_ids'])
		## to reshape conversations
		max_num_utterances_batch = batch['max_num_utterances']
		return batch_size, goldids_variable, conversation_mask, elmo_embeddings, input_mask_variable, \
				sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch,\
			   options_tensor, goldids_variable




	def prepare_for_cpu(self, *input):
		loss = input[0]
		indices = input[1]
		if self.args.use_cuda:
			indices = indices.data.cpu()
		else:
			indices = indices.data
		if self.args.use_cuda:
			loss = loss.data.cpu()
		else:
			loss = loss.data
		return loss, indices

