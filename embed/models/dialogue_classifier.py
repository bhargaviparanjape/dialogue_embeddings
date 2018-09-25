import torch
import torch.nn as nn
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor
from embed.models import factory as model_factory
import numpy as np
from torch.nn import functional
import pdb


@RegisterModel('dialogue_classifier')
class DialogueClassifier(nn.Module):
	def __init__(self, args):
		super(DialogueClassifier, self).__init__()
		self.args = args

		self.lookup_layer = model_factory.get_model(args, args.lookup)
		self.encoding_layer = model_factory.get_model(args, args.encoding)

		self.next_utterance_scorer = nn.Bilinear(2 * args.encoder_hidden_size, 2 * args.encoder_hidden_size, 1)
		self.prev_utterance_scorer = nn.Bilinear(2 * args.encoder_hidden_size, 2 * args.encoder_hidden_size, 1)

		# self.next_utterance_scorer = nn.Bilinear(2 * args.encoder_num_layers * args.encoder_hidden_size,
		# 										 2 * args.encoder_num_layers * args.encoder_hidden_size, 1)
		# self.prev_utterance_scorer = nn.Bilinear(2 * args.encoder_num_layers * args.encoder_hidden_size,
		# 										 2 * args.encoder_num_layers * args.encoder_hidden_size, 1)

	def forward(self, *input):
		[embeddings, input_mask_variable, \
		 sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch, \
		 options_tensor, goldids_next_variable, goldids_prev_variable, labels] = input[0]

		if self.args.embedding != "avg_elmo":
			lookup = self.lookup_layer(embeddings, input_mask_variable)
		else:
			lookup = embeddings
		## reshape based on max_input_length in batch dimension
		sequence_batch_size = int(lookup.shape[0] / max_num_utterances_batch)
		reshaped_lookup = lookup.view(sequence_batch_size, max_num_utterances_batch, lookup.shape[1])
		## sort utterances then apply encoding layer
		sorted_lookup = reshaped_lookup[sort]


		## get hidden representations
		encoded, _ = self.encoding_layer(sorted_lookup, lengths_sorted)
		# encoded = self.encoding_layer(sorted_lookup, conversation_mask_sorted)
		# encoded = torch.cat((encoded[0,:], encoded[1,:]), 2)


		encoded = encoded[unsort].view(sequence_batch_size * max_num_utterances_batch, -1, encoded.shape[2])
		## do lookup based on indices
		options = torch.index_select(encoded.squeeze(1), 0, options_tensor.view(-1))
		## expand the options K times get
		encoded_expand = encoded.expand(encoded.shape[0], options_tensor.shape[1], encoded.shape[2]).contiguous()
		encoded_expand = encoded_expand.view(encoded_expand.shape[0] * options_tensor.shape[1], encoded_expand.shape[2])
		## MLP
		next_logits = self.next_utterance_scorer(encoded_expand, options)
		prev_logits = self.prev_utterance_scorer(encoded_expand, options)
		next_logits_flat = next_logits.view(encoded.shape[0], options_tensor.shape[1], -1)
		next_log_probs_flat = functional.log_softmax(next_logits_flat)
		prev_logits_flat = prev_logits.view(encoded.shape[0], options_tensor.shape[1], -1)
		prev_log_probs_flat = functional.log_softmax(prev_logits_flat)
		conversation_mask = conversation_mask_sorted[unsort]
		losses_flat = -torch.gather(next_log_probs_flat.squeeze(2), dim=1, index=goldids_next_variable.view(-1, 1)) \
			+ (-torch.gather(prev_log_probs_flat.squeeze(2), dim=1, index=goldids_prev_variable.view(-1, 1)))
		losses = losses_flat * conversation_mask.view(sequence_batch_size*max_num_utterances_batch, -1)
		## loss and indices (average next and previous prediction answers)
		loss = losses.sum() / (2*conversation_mask.float().sum())
		next_predictions = torch.sort(next_logits_flat.squeeze(2), descending=True)[1][:, 0]
		prev_predictions = torch.sort(prev_logits_flat.squeeze(2), descending=True)[1][:, 0]

		return loss, tuple([next_predictions, prev_predictions])

	def dump_embeddings(self, *input):
		[embeddings, input_mask_variable, \
		 sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch, \
		 options_tensor, goldids_next_variable, goldids_prev_variable, labels] = input[0]
		if self.args.embedding != "avg_elmo":
			lookup = self.lookup_layer(embeddings, input_mask_variable)
		else:
			lookup = embeddings
		sequence_batch_size = int(lookup.shape[0] / max_num_utterances_batch)
		reshaped_lookup = lookup.view(sequence_batch_size, max_num_utterances_batch, lookup.shape[1])
		sorted_lookup = reshaped_lookup[sort]


		## get hidden representations
		encoded, _ = self.encoding_layer(sorted_lookup, lengths_sorted)
		# encoded = self.encoding_layer(sorted_lookup, conversation_mask_sorted)
		# encoded = torch.cat((encoded[0, :], encoded[1, :]), 2)


		encoded = encoded[unsort].view(sequence_batch_size * max_num_utterances_batch, -1, encoded.shape[2])
		return encoded.squeeze(1)

	def eval(self, *input):
		[embeddings, input_mask_variable, \
		 sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch, \
		 options_tensor, goldids_next_variable, goldids_prev_variable, labels] = input[0]
		if self.args.embedding != "avg_elmo":
			lookup = self.lookup_layer(embeddings, input_mask_variable)
		else:
			lookup = embeddings
		## reshape based on max_input_length in batch dimension
		sequence_batch_size = int(lookup.shape[0] / max_num_utterances_batch)
		reshaped_lookup = lookup.view(sequence_batch_size, max_num_utterances_batch, lookup.shape[1])
		## sort utterances then apply encoding layer
		sorted_lookup = reshaped_lookup[sort]


		## get hidden representations
		encoded, _ = self.encoding_layer(sorted_lookup, lengths_sorted)
		# encoded = self.encoding_layer(sorted_lookup, conversation_mask_sorted)
		# encoded = torch.cat((encoded[0, :], encoded[1, :]), 2)

		encoded = encoded[unsort].view(sequence_batch_size * max_num_utterances_batch, -1, encoded.shape[2])
		## do lookup based on indices
		options = torch.index_select(encoded.squeeze(1), 0, options_tensor.view(-1))
		## expand the options K times get
		encoded_expand = encoded.expand(encoded.shape[0], options_tensor.shape[1], encoded.shape[2]).contiguous()
		encoded_expand = encoded_expand.view(encoded_expand.shape[0] * options_tensor.shape[1], encoded_expand.shape[2])
		## MLP
		next_logits = self.next_utterance_scorer(encoded_expand, options)
		prev_logits = self.prev_utterance_scorer(encoded_expand, options)
		next_logits_flat = next_logits.view(encoded.shape[0], options_tensor.shape[1], -1)
		prev_logits_flat = prev_logits.view(encoded.shape[0], options_tensor.shape[1], -1)
		next_predictions = torch.sort(next_logits_flat.squeeze(2), descending=True)[1][:, 0]
		prev_predictions = torch.sort(prev_logits_flat.squeeze(2), descending=True)[1][:, 0]


		return tuple([next_predictions, prev_predictions])

	def prepare_for_gpu(self, batch, embedding_layer):
		batch_size = int(len(batch['utterance_list']) / batch['max_num_utterances'])
		max_num_utterances_batch = batch['max_num_utterances']

		### Prepare embeddings
		if self.args.embedding == "elmo":
			### embedding_layer doesnt recide on cuda if its elmo
			utterance_embeddings, input_mask = embedding_layer.get_embeddings(batch['utterance_list'])
			embedding_layer.get_embeddings()

		elif self.args.embedding == "glove":
			## change the batch into LongTensor
			batch_ids = LongTensor(batch["utterance_word_ids"])
			utterance_embeddings = embedding_layer.lookup(batch_ids)
			input_mask = FloatTensor(batch['input_mask'])

		elif self.args.embedding == "avg_elmo":
			conversation_ids = batch["conversation_ids"]
			utterance_embeddings = embedding_layer.lookup(conversation_ids, max_num_utterances_batch)
			input_mask = FloatTensor(batch["input_mask"])

		if self.args.use_cuda:
			utterance_embeddings = utterance_embeddings.cuda()
		input_mask_variable = variable(input_mask)

		### Prepare Encoder layer
		batch_lengths = batch['conversation_lengths']
		length_sort = np.argsort(batch_lengths)[::-1].copy()
		unsort = variable(LongTensor(np.argsort(length_sort)))
		conversation_mask_sorted = variable(FloatTensor(batch['conversation_mask'])[length_sort])
		conversation_mask = LongTensor(batch['conversation_mask'])
		lengths_sorted = np.array(batch_lengths)[length_sort]
		sort = length_sort

		### Prepare output layer
		options_tensor = LongTensor(batch['utterance_options_list'])
		goldids_next_variable = LongTensor(batch['next_utterance_gold'])
		goldids_prev_variable = LongTensor(batch['prev_utterance_gold'])
		utterance_labels = LongTensor(batch['label'])

		return batch_size, tuple([goldids_next_variable, goldids_prev_variable]), conversation_mask, utterance_embeddings, input_mask_variable, \
			   sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch, \
			   options_tensor, goldids_next_variable, goldids_prev_variable, utterance_labels

	def prepare_for_cpu(self, loss , *input):
		next_indices = input[0][0]
		prev_indices = input[0][1]
		if self.args.use_cuda:
			next_indices = next_indices.data.cpu()
			prev_indices = prev_indices.data.cpu()
			loss = loss.data.cpu()
		else:
			next_indices = next_indices.data
			prev_indices = prev_indices.data
			loss = loss.data
		return loss, next_indices, prev_indices

