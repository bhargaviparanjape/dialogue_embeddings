import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor
from embed.models import factory as model_factory
from embed.models.dialogue_embedder import DialogueEmbedder
import numpy as np
from torch.nn import functional


@RegisterModel('dialogue_bow')
class DialogueBagOfWordClassifier(nn.Module):
	def __init__(self, args):
		super(DialogueBagOfWordClassifier, self).__init__()
		self.args = args

		self.dialogue_embedder = DialogueEmbedder(args)

		if args.encoding == "bilstm":
			hidden_size = 2 * args.encoder_hidden_size
		else:
			hidden_size = 2 * args.encoder_hidden_size * args.encoder_num_layers

		self.bow_scorer = nn.Sequential(
			nn.Linear(hidden_size, 2048),
			nn.ReLU(),
			nn.Linear(2048, len(args.vocabulary))
		)
		# self.masked_softmax = MaskedSoftmaxAndLogSoftmax()

		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, 100),
			nn.ReLU(),
			nn.Linear(100, self.args.output_size))
		self.classifier_loss = torch.nn.CrossEntropyLoss()

	def masked_softmax(self, input, target, mask):
		'''
		maxes = torch.max(input + torch.log(mask), 1, keepdim=True)[0]
		masked_exp_xs = torch.exp(input - maxes) * mask
		#masked_exp_xs[masked_exp_xs != masked_exp_xs] = 0
		normalization_factor = masked_exp_xs.sum(1, keepdim=True)
		# probs = masked_exp_xs / normalization_factor
		score_log_probs = (input - maxes - torch.log(normalization_factor)) * mask
		#score_log_probs[score_log_probs != score_log_probs] = 0
		loss = (-(score_log_probs * mask)).sum() / mask.sum()
		'''
		negative_log_prob = -(F.log_softmax(input))
		loss = (torch.gather(negative_log_prob, 1, target)*mask).sum()/ mask.sum()
		return loss

	def forward(self, *input):
		[embeddings, input_mask_variable, \
		 conversation_mask, max_num_utterances_batch, \
		 max_utterance_length, utterance_word_ids, labels] = input[0]

		encoded = self.dialogue_embedder(input[0][:4])

		sequence_batch_size = int(embeddings.shape[0] / max_num_utterances_batch)

		## dialogue auxiliary task
		vocabulary_scores = self.bow_scorer(encoded.squeeze(1))
		# batch_token_scores = torch.gather(vocabulary_scores, 1, utterance_word_ids)

		# score_probs, score_log_probs = self.masked_softmax(batch_token_scores, input_mask_variable)

		loss = self.masked_softmax(vocabulary_scores, utterance_word_ids, input_mask_variable)

		## for each word sample the top utterances
		predictions = torch.sort(vocabulary_scores, descending=True)[1][:, :max_utterance_length]

		label_logits = self.classifier(encoded.squeeze(1))
		label_log_probs_flat = functional.log_softmax(label_logits, dim=1)
		label_losses_flat = -torch.gather(label_log_probs_flat, dim=1, index=labels.view(-1, 1))
		label_losses = label_losses_flat * conversation_mask.view(sequence_batch_size * max_num_utterances_batch, -1)
		label_loss = label_losses.sum() / conversation_mask.float().sum()
		labels_predictions = torch.sort(label_logits, descending=True)[1][:, 0]

		combined_loss = (0.7*loss + 0.3*label_loss)

		return combined_loss, tuple([predictions, labels_predictions])

	def eval(self, *input):
		[embeddings, input_mask_variable, \
		 conversation_mask, max_num_utterances_batch, \
		 max_utterance_length, utterance_word_ids, labels] = input[0]

		encoded = self.dialogue_embedder(input[0][:4])

		sequence_batch_size = int(embeddings.shape[0] / max_num_utterances_batch)

		## dialogue auxiliary task
		vocabulary_scores = self.bow_scorer(encoded.squeeze(1))
		## for each word sample the top utterances
		predictions = torch.sort(vocabulary_scores, descending=True)[1][:, :max_utterance_length]


		## predict dialogue act labels
		label_logits = self.classifier(encoded.squeeze(1))
		labels_predictions = torch.sort(label_logits, descending=True)[1][:, 0]

		return tuple([predictions, labels_predictions])

	def prepare_for_gpu(self, batch, embedding_layer):
		batch_size = int(len(batch['utterance_list']) / batch['max_num_utterances'])
		max_num_utterances_batch = batch['max_num_utterances']
		max_utterance_length = batch['max_utterance_length']
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
			batch_ids = LongTensor(batch["utterance_word_ids"])
			conversation_ids = batch["conversation_ids"]
			utterance_embeddings = embedding_layer.lookup(conversation_ids, max_num_utterances_batch)
			input_mask = FloatTensor(batch['input_mask'])

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

		return batch_size, tuple([batch_ids,
								  utterance_labels]), input_mask, utterance_embeddings, input_mask_variable, \
			   variable(conversation_mask.float()), max_num_utterances_batch, \
			   max_utterance_length, batch_ids, utterance_labels \

	def prepare_for_cpu(self, loss, *input):
		word_indices = input[0][0]
		label_indices = input[0][1]
		if self.args.use_cuda:
			word_indices = word_indices.data.cpu()
			label_indices = label_indices.data.cpu()
			loss = loss.data.cpu()
		else:
			word_indices = word_indices.data
			label_indices = label_indices.data
			loss = loss.data
		return loss, word_indices, label_indices
