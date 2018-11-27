import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import pdb
import copy,os,logging

from src.models.abstract_model import AbstractModel
from src.models import factory as model_factory
from src.learn import factory as learn_factory
from src.models.factory import RegisterModel
from src.models.components.output_models.dialogue_embedder import DialogueEmbedder
from src.utils.utility_functions import variable,FloatTensor,ByteTensor,LongTensor,select_optimizer

logger = logging.getLogger(__name__)


#########################################
############### NETWORK #################
#########################################
@RegisterModel('da_bow_classifier_network')
class DialogueActBowClassifierNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueActBowClassifierNetwork, self).__init__()

		# copy args for bow, replace utterene embedding layer to average, initialze
		pretrained_args = copy.deepcopy(args)
		vars(args)["utterance-encoder"] = "avg"
		# For average Elmo, even fixed-utterance-encoder needs to be reset
		self.dialogue_embedder = DialogueEmbedder(pretrained_args)

		# dialogue embedder for the classifier
		vars(args)["utterance-encoder"] = "recurrent"
		self.dialogue_embedder_for_classifier = DialogueEmbedder(args)

		## Define class network
		## output labels size
		dict_ = {"input_size": args.output_input_size, "hidden_size": args.output_hidden_size, "num_layers" : args.output_num_layers,
							 "output_size": args.output_size}
		self.classifier = model_factory.get_model_by_name(args.output_layer, args, kwargs = dict_)
		self.args = args

		if self.args.objective == "linear_crf":
			dict_ = {"output_size": args.output_size}
			self.structured_layer = model_factory.get_model_by_name(args.objective, args, kwargs = dict_)

	def label_cross_entropy(self, label_logits, labels_flattened, mask_flattened):
		label_log_probs_flat = F.log_softmax(label_logits, dim=1)
		label_losses_flat = -torch.gather(label_log_probs_flat, dim=1, index=labels_flattened.view(-1, 1))
		label_losses = label_losses_flat * mask_flattened
		loss = label_losses.sum() / mask_flattened.float().sum()
		return loss

	def linear_crf(self, logits_flattened, labels_flattened, mask_flattened, h, w):
		logits = logits_flattened.view(h,w,-1)
		labels = labels_flattened.view(h,w,-1)
		mask = mask_flattened.view(h, w)
		return self.structured_layer(logits, labels, mask)

	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch , gold_labels] = input

		conversation_encoded, utterance_encodings = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_encoded_classifier, utterance_encodings = self.dialogue_embedder_for_classifier(
			[token_embeddings, input_mask_variable, conversation_mask,
			 max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		# Rejoin both directions
		conversation_encoded = conversation_encoded.view(conversation_encoded.shape[0], 1, -1).squeeze(1)
		conversation_encoded_classifier = conversation_encoded_classifier.view(conversation_encoded_classifier.shape[0], 1, -1).squeeze(1)

		label_logits = self.classifier(torch.cat((conversation_encoded, conversation_encoded_classifier), 1))
		new_batch_size = conversation_mask.shape[0]
		new_conversation_size = conversation_mask.shape[1]

		if self.args.objective == "linear_crf":
			loss = self.linear_crf(label_logits, gold_labels, conversation_mask, new_batch_size, new_conversation_size)
		else:
			loss = self.label_cross_entropy(label_logits, gold_labels, conversation_mask)


		return loss

	def evaluate(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch] = input

		conversation_encoded, utterance_encodings = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_encoded_classifier, utterance_encodings = self.dialogue_embedder_for_classifier(
			[token_embeddings, input_mask_variable, conversation_mask,
			 max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		# Rejoin both directions
		conversation_encoded = conversation_encoded.view(conversation_encoded.shape[0], 1, -1).squeeze(1)
		conversation_encoded_classifier = conversation_encoded_classifier.view(conversation_encoded_classifier.shape[0], 1, -1).squeeze(1)

		label_logits = self.classifier(torch.cat((conversation_encoded, conversation_encoded_classifier), 1))


		## when doing CRF inference; do viterbi decoding
		h, w = conversation_batch_size, max_num_utterances_batch
		if self.args.objective == "linear_crf":
			best_paths = self.structured_layer._viterbi_decode(label_logits.view(h, w, -1), conversation_mask)
			best_path_ids = [b[0] for b in best_paths]
			return best_path_ids
		return label_logits

	@staticmethod
	def add_args(parser):
		pass


#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('da_bow_classifier')
class DialogueActBowClassifier(AbstractModel):
	def __init__(self, args):

		## Initialize environment
		self.args = args
		self.updates = 0

		## If token encodings are not computed on the fly using character CNN based models but are obtained from a pretrained model
		if args.fixed_token_encoder:
			self.token_encoder = model_factory.get_embeddings(args.token_encoder, args)

		self.network = model_factory.get_model_by_name(args.network, args)

		## TODO: Set embedding layer parameters trainable or tunable

	def predict(self, inputs):
		# Eval mode
		self.network.eval()

		# Run forward
		batch_size, *inputs = self.vectorize(inputs, mode = "test")
		scores_logits = self.network.evaluate(*inputs)
		if self.args.objective == "linear_crf":
			labels_predictions = torch.LongTensor(scores_logits).flatten()
			if self.args.use_cuda:
				input_mask = inputs[2].data.cpu()
			else:
				input_mask = inputs[2].data
		else:
			labels_predictions = torch.sort(scores_logits, descending=True)[1][:, 0]
			# Convert to CPU
			if self.args.use_cuda:
				labels_predictions = labels_predictions.data.cpu()
				input_mask = inputs[2].data.cpu()
			else:
				labels_predictions = labels_predictions.data
				input_mask = inputs[2].data

		# Mask inputs
		return [labels_predictions], input_mask


	def target(self, inputs):
		batch_size, *inputs = self.vectorize(inputs, mode="train")
		# Convert to CPU
		if self.args.use_cuda:
			true_labels = inputs[-1].data.cpu()
			input_mask = inputs[2].data.cpu()
		else:
			true_labels = inputs[-1].data
			input_mask = inputs[2].data
		return [true_labels], input_mask

	def evaluate_metrics(self, predicted, target, mask, mode = "dev"):
		# Named Metric List
		batch_size = mask.shape[0]
		# mask = mask[:, 2:].contiguous()
		mask = mask.view(-1, 1).squeeze(1)
		label_predicted = predicted[0]
		label_correct = target[0].view(batch_size, -1).view(label_predicted.shape) #[:, 1:-1].contiguous().view(label_predicted.shape)
		predictions_binary = (label_predicted == label_correct)
		correct = (predictions_binary.long()*mask.long()).sum().numpy()
		total = mask.sum().data.numpy()
		metric_update_dict = {}
		metric_update_dict[self.args.metric[0]] = [correct, total]
		return metric_update_dict

	def set_vocabulary(self, vocabulary):
		self.vocabulary = vocabulary
		## Embedding layer initialization depends upon vocabulary
		if hasattr(self.token_encoder, "load_embeddings"):
			self.token_encoder.load_embeddings(self.vocabulary)

	def vectorize(self, batch, mode = "train"):
		## TODO: Get single example, abstract out batchification
		batch_size = int(len(batch['utterance_list']) / batch['max_num_utterances'])
		max_num_utterances_batch = batch['max_num_utterances']

		## Prepare Token Embeddings
		token_embeddings, token_mask = self.token_encoder.lookup(batch)
		if self.args.use_cuda:
			token_embeddings = token_embeddings.cuda()
		input_mask_variable = variable(token_mask)

		conversation_lengths = batch['conversation_lengths']
		conversation_mask = variable(FloatTensor(batch['conversation_mask']))

		## Prepare Ouput (If exists)
		utterance_labels = LongTensor(batch['label'])

		if mode == "train":
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
				utterance_labels
		else:
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch

	@staticmethod
	def load(filename, new_args=None, normalize=True):
		logger.info('Loading model %s' % filename)
		saved_params = torch.load(
			filename, map_location=lambda storage, loc: storage
		)
		word_dict = saved_params['word_dict']
		state_dict = saved_params['state_dict']
		args = new_args
		# TODO: Handle code for replaceing new paramters with older ones
		# Not handling fixed embedding layer
		model = DialogueActBowClassifier(args)
		model.network.load_state_dict(state_dict, strict=False)
		model.vocabulary = word_dict
		return model

	@staticmethod
	def add_args(parser):
		pass