import torch
import torch.nn as nn
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor
from embed.models import factory as model_factory
import numpy as np
from torch.nn import functional
from embed.models.dialogue_embedder import DialogueEmbedder


@RegisterModel('dialogue_act_classifier')
class DialogueActClassifier(nn.Module):
	def __init__(self, args):
		super(DialogueActClassifier, self).__init__()
		self.args = args

		self.dialogue_embedder = DialogueEmbedder(args)

		if args.encoding == "bilstm":
			hidden_size = 2 * args.encoder_hidden_size
		else:
			hidden_size = 2 * args.encoder_hidden_size * args.encoder_num_layers
		self.next_utterance_scorer = nn.Bilinear(hidden_size, hidden_size, 1)
		self.prev_utterance_scorer = nn.Bilinear(hidden_size,hidden_size, 1)

		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, 100),
			nn.ReLU(),
			nn.Linear(100, self.args.output_size))
		self.classifier_loss = torch.nn.CrossEntropyLoss()

	def masked_softmax(self, input, mask):
		maxes = torch.max(input + torch.log(mask), 1, keepdim=True)[0]
		masked_exp_xs = torch.exp(input - maxes) * mask
		masked_exp_xs[masked_exp_xs != masked_exp_xs] = 0
		normalization_factor = masked_exp_xs.sum(1, keepdim=True)
		# probs = masked_exp_xs / normalization_factor
		score_log_probs = (input - maxes - torch.log(normalization_factor)) * mask
		score_log_probs[score_log_probs != score_log_probs] = 0
		loss = (-(score_log_probs * mask)).sum() / mask.sum()
		return loss

	def forward(self, *input):
		[embeddings, input_mask_variable, \
		 conversation_mask, max_num_utterances_batch, \
		 options_tensor, goldids_next_variable, goldids_prev_variable, labels] = input[0]

		encoded = self.dialogue_embedder(input[0][:4])

		sequence_batch_size = int(embeddings.shape[0] / max_num_utterances_batch)

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
		losses_flat = -torch.gather(next_log_probs_flat.squeeze(2), dim=1, index=goldids_next_variable.view(-1, 1)) \
					  + (-torch.gather(prev_log_probs_flat.squeeze(2), dim=1, index=goldids_prev_variable.view(-1, 1)))
		losses = losses_flat * conversation_mask.view(sequence_batch_size * max_num_utterances_batch, -1)
		## loss and indices (average next and previous prediction answers)
		loss = losses.sum() / (2 * conversation_mask.float().sum())
		next_predictions = torch.sort(next_logits_flat.squeeze(2), descending=True)[1][:, 0]
		prev_predictions = torch.sort(prev_logits_flat.squeeze(2), descending=True)[1][:, 0]

		label_logits = self.classifier(encoded.squeeze(1))
		label_log_probs_flat = functional.log_softmax(label_logits, dim=1)
		label_losses_flat = -torch.gather(label_log_probs_flat, dim=1, index=labels.view(-1, 1))
		label_losses = label_losses_flat * conversation_mask.view(sequence_batch_size * max_num_utterances_batch, -1)
		label_loss = label_losses.sum()/conversation_mask.float().sum()
		labels_predictions = torch.sort(label_logits, descending=True)[1][:, 0]
		combined_loss = (loss + label_loss)/2
		return combined_loss, tuple([next_predictions, prev_predictions, labels_predictions])

	def eval(self, *input):
		[embeddings, input_mask_variable, \
		 conversational_mask, max_num_utterances_batch, \
		 options_tensor, goldids_next_variable, goldids_prev_variable, labels] = input[0]

		encoded = self.dialogue_embedder(input[0][:4])

		sequence_batch_size = int(embeddings.shape[0] / max_num_utterances_batch)

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

		## predict dialogue act labels
		label_logits = self.classifier(encoded.squeeze(1))
		labels_predictions = torch.sort(label_logits, descending=True)[1][:, 0]

		return tuple([next_predictions, prev_predictions, labels_predictions])

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

		return batch_size, tuple([goldids_next_variable,
			goldids_prev_variable, utterance_labels]), conversation_mask, utterance_embeddings, input_mask_variable, \
			variable(conversation_mask.float()), max_num_utterances_batch, \
			options_tensor, goldids_next_variable, goldids_prev_variable, utterance_labels

	def prepare_for_cpu(self, loss, *input):
		next_indices = input[0][0]
		prev_indices = input[0][1]
		label_indices = input[0][2]
		if self.args.use_cuda:
			next_indices = next_indices.data.cpu()
			prev_indices = prev_indices.data.cpu()
			label_indices = label_indices.data.cpu()
			loss = loss.data.cpu()
		else:
			next_indices = next_indices.data
			prev_indices = prev_indices.data
			label_indices = label_indices.data
			loss = loss.data
		return loss, next_indices, prev_indices, label_indices
