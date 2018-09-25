import torch
import torch.nn as nn
import torch.autograd as ag
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor
from embed.models import factory as model_factory
import numpy as np
from torch.nn import functional


class MaskedSoftmaxAndLogSoftmax(ag.Function):
  def __init__(self, dtype = torch.FloatTensor):
    super(MaskedSoftmaxAndLogSoftmax, self).__init__()
    self._dtype = dtype

  def forward(self, xs, mask):
    maxes = torch.max(xs + torch.log(mask), 1, keepdim = True)[0]
    masked_exp_xs = torch.exp(xs - maxes) * mask
    normalization_factor = masked_exp_xs.sum(1, keepdim = True)
    probs = masked_exp_xs / normalization_factor
    log_probs = (xs - maxes - torch.log(normalization_factor)) * mask

    self.save_for_backward(probs, mask)
    return probs, log_probs

  def backward(self, grad_probs, grad_log_probs):
    probs, mask = self.saved_tensors

    num_actions = grad_probs.size()[1]
    w1 = (probs * grad_probs).unsqueeze(0).unsqueeze(-1)
    w2 = torch.eye(num_actions).type(self._dtype).unsqueeze(0)
    if grad_probs.is_cuda:
      w2 = w2.cuda()
    w2 = (w2 - probs.unsqueeze(-1))

    grad1 = torch.matmul(w2, w1).squeeze(0).squeeze(-1)

    w1 = grad_log_probs
    sw1 = (mask * grad_log_probs).sum(1, keepdim = True)
    grad2 = (w1 * mask - probs * sw1)
    return grad1 + grad2, None


@RegisterModel('dialogue_bow')
class DialogueBagOfWordClassifier(nn.Module):
	def __init__(self, args):
		super(DialogueBagOfWordClassifier, self).__init__()
		self.args = args

		self.lookup_layer = model_factory.get_model(args, args.lookup)
		self.encoding_layer = model_factory.get_model(args, args.encoding)

		self.next_utterance_scorer = nn.Bilinear(2 * args.encoder_hidden_size, 2 * args.encoder_hidden_size, 1)
		self.prev_utterance_scorer = nn.Bilinear(2 * args.encoder_hidden_size, 2 * args.encoder_hidden_size, 1)

		self.bow_scorer = nn.Sequential(
			nn.Linear(2 * args.encoder_hidden_size, 2048),
			nn.ReLU(),
			nn.Linear(2048, len(args.vocabulary))
		)
		self.masked_softmax = MaskedSoftmaxAndLogSoftmax()

		self.classifier = nn.Sequential(
			nn.Linear(2 * args.encoder_hidden_size, 100),
			nn.ReLU(),
			nn.Linear(100, self.args.output_size))
		self.classifier_loss = torch.nn.CrossEntropyLoss()


	def forward(self, *input):
		[embeddings, input_mask_variable, \
		 sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch, \
		 max_utterance_length, utterance_word_ids, labels] = input[0]

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
		encoded = encoded[unsort].view(sequence_batch_size * max_num_utterances_batch, -1, encoded.shape[2])

		## dialogue auxiliary task
		vocabulary_scores = self.bow_scorer(encoded.squeeze(1))
		batch_token_scores = torch.gather(vocabulary_scores, 1, utterance_word_ids)
		score_probs, score_log_probs = self.masked_softmax(batch_token_scores, input_mask_variable)
		loss = (-(score_log_probs*input_mask_variable)).sum()/ input_mask_variable.sum()
		## for each word sample the top utterances
		predictions = torch.sort(vocabulary_scores, descending=True)[1][:, :max_utterance_length]

		label_logits = self.classifier(encoded.squeeze(1))
		## Not including DA prediction loss as part of the objective
		labels_predictions = torch.sort(label_logits, descending=True)[1][:, 0]

		return loss, tuple([predictions, labels_predictions])

	def eval(self, *input):
		[embeddings, input_mask_variable, \
		 sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch, \
		 max_utterance_length, utterance_word_ids, labels] = input[0]

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
		encoded = encoded[unsort].view(sequence_batch_size * max_num_utterances_batch, -1, encoded.shape[2])

		## dialogue auxiliary task
		vocabulary_scores = self.bow_scorer(encoded.squeeze(1))
		## for each word sample the top utterances
		predictions = torch.sort(vocabulary_scores, descending=True)[1][:, max_utterance_length]


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
			conversation_ids = batch["conversation_ids"]
			embedding_layer.lookup(conversation_ids, max_num_utterances_batch)

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
			   sort, unsort, conversation_mask_sorted, lengths_sorted, max_num_utterances_batch, \
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