import torch
import torch.nn as nn
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor
from embed.models import factory as model_factory
import numpy as np
from torch.nn import functional
from embed.models.dialogue_embedder import DialogueEmbedder

@RegisterModel('da_model')
class DAClassififier(nn.Module):
	def __init__(self, args):
		super(DAClassififier, self).__init__()
		self.args = args

		self.dialogue_embedder = DialogueEmbedder(args)

		if args.encoding == "bilstm":
			hidden_size = 2 * args.encoder_hidden_size
		else:
			hidden_size = 2 * args.encoder_hidden_size * args.encoder_num_layers

		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, 100),
			nn.ReLU(),
			nn.Linear(100, self.args.output_size))
		self.classifier_loss = torch.nn.CrossEntropyLoss()


	def forward(self, *input):
		[embeddings, input_mask_variable, \
		 conversation_mask, max_num_utterances_batch, \
		 labels] = input[0]

		encoded = self.dialogue_embedder(input[0][:4])

		sequence_batch_size = int(embeddings.shape[0] / max_num_utterances_batch)

		label_logits = self.classifier(encoded.squeeze(1))
		label_log_probs_flat = functional.log_softmax(label_logits, dim=1)
		label_losses_flat = -torch.gather(label_log_probs_flat, dim=1, index=labels.view(-1, 1))
		label_losses = label_losses_flat * conversation_mask.view(sequence_batch_size * max_num_utterances_batch, -1)
		label_loss = label_losses.sum() / conversation_mask.float().sum()
		labels_predictions = torch.sort(label_logits, descending=True)[1][:, 0]
		return label_loss, labels_predictions


	def eval(self, *input):
		[embeddings, input_mask_variable, \
		 conversation_mask, max_num_utterances_batch, \
		 labels] = input[0]

		encoded = self.dialogue_embedder(input[0][:4])

		sequence_batch_size = int(embeddings.shape[0] / max_num_utterances_batch)

		label_logits = self.classifier(encoded.squeeze(1))
		labels_predictions = torch.sort(label_logits, descending=True)[1][:, 0]
		return labels_predictions


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
		utterance_labels = LongTensor(batch['label'])

		return batch_size, utterance_labels, conversation_mask, utterance_embeddings, input_mask_variable, \
			   variable(conversation_mask.float()), max_num_utterances_batch, \
			   utterance_labels

	def prepare_for_cpu(self, loss, *input):
		label_indices = input[0]
		if self.args.use_cuda:
			label_indices = label_indices.data.cpu()
			loss = loss.data.cpu()
		else:
			label_indices = label_indices.data
			loss = loss.data
		return loss, label_indices



