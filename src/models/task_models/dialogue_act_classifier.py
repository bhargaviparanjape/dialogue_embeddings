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
@RegisterModel('da_classifier_network1')
class DialogueActClassifierNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueActClassifierNetwork, self).__init__()
		self.dialogue_embedder = DialogueEmbedder(args)

		## Define class network
		## output labels size
		dict_ = {"input_size": args.output_input_size, "hidden_size": args.output_hidden_size[0], "num_layers" : args.output_num_layers[0],
							 "output_size": args.output_size}
		self.classifier = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)

		## Define loss function: Custom masked entropy


	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch , gold_labels] = input

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		label_logits = self.classifier(conversation_encoded.squeeze(1))
		label_log_probs_flat = F.log_softmax(label_logits, dim=1)
		label_losses_flat = -torch.gather(label_log_probs_flat, dim=1, index=gold_labels.view(-1, 1))
		label_losses = label_losses_flat * conversation_mask.view(conversation_batch_size * max_num_utterances_batch, -1)
		loss = label_losses.sum() / conversation_mask.float().sum()
		return loss

	def evaluate(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch] = input
		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		label_logits = self.classifier(conversation_encoded.squeeze(1))
		return label_logits

	@staticmethod
	def add_args(parser):
		pass
		# model_parameters = parser.add_argument_group("Model Parameters")
		# model_parameters.add_argument("--output-size", type=int)
		# model_parameters.add_argument("--output-hidden-size", type=int, action="append")
		# model_parameters.add_argument("--output-input-size", type=int)
		# model_parameters.add_argument("--output-num-layers", type=int , action="append")


#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('da_classifier1')
class DialogueActClassifier(AbstractModel):
	def __init__(self, args):

		## Initialize environment
		self.args = args
		self.updates = 0

		## If token encodings are not computed on the fly using character CNN based models but are obtained from a pretrained model
		if args.fixed_token_encoder:
			self.token_encoder = model_factory.get_embeddings(args.token_encoder, args)

		self.network = model_factory.get_model_by_name(args.network, args)

		## Set embedding layer parameters trainable or tunable

	def cuda(self):
		self.network = self.network.cuda()

	def update(self, inputs):
		## update based on inputs
		"""Forward a batch 	of examples; step the optimizer to update weights."""
		if not self.optimizer:
			raise RuntimeError('No optimizer set.')

		# Train mode
		self.network.train()

		# Run forward
		batch_size, *inputs = self.vectorize(inputs, mode = "train")
		loss = self.network(*inputs)

		# Update parameters
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.network.parameters(),
									   self.args.clip_threshold)
		self.optimizer.step()
		self.updates += 1

		# Return loss and batch size [to average over]
		if self.args.use_cuda:
			loss_value = loss.data.cpu().item()
		else:
			loss_value = loss.data.item()
		return loss_value, batch_size

	def checkpoint(self, file_path, epoch_no):
		raise NotImplementedError

	def predict(self, inputs):
		# Eval mode
		self.network.eval()

		# Run forward
		batch_size, *inputs = self.vectorize(inputs, mode = "test")
		scores_logits = self.network.evaluate(*inputs)
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
		mask = mask.view(-1, 1).squeeze(1)
		label_predicted = predicted[0]
		label_correct = target[0]
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

		## Prepare Utterance Encoder

		## Prepare Conversation Encoder
		## TODO: Abstraction similar to token embeddings
		conversation_lengths = batch['conversation_lengths']
		conversation_mask = variable(FloatTensor(batch['conversation_mask']))

		## Prepare Ouput (If exists)
		utterance_labels = LongTensor(batch['label'])

		if mode == "train":
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
				utterance_labels
		else:
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch


	def init_optimizer(self):
		parameters = [p for p in self.network.parameters() if p.requires_grad]
		self.optimizer = select_optimizer(self.args, parameters)


	def parallelize(self):
		"""Use data parallel to copy the model across several gpus.
		This will take all gpus visible with CUDA_VISIBLE_DEVICES.
		"""
		self.parallel = True
		self.network = torch.nn.DataParallel(self.network)


	def save(self):
		# model parameters; metrics;
		if self.args.parallel:
			network = self.network.module
		else:
			network = self.network
		state_dict = copy.copy(network.state_dict())
		# Pop layers if required
		params = {
			'word_dict': self.vocabulary,
			'args': self.args,
			'state_dict': state_dict
		}
		try:
			torch.save(params, os.path.join(self.args.model_dir, self.args.model_path))
		except BaseException:
			logger.warning('WARN: Saving failed... continuing anyway.')

	@staticmethod
	def load(filename, new_args=None, normalize=True):
		logger.info('Loading model %s' % filename)
		saved_params = torch.load(
			filename, map_location=lambda storage, loc: storage
		)
		word_dict = saved_params['word_dict']
		state_dict = saved_params['state_dict']
		args = saved_params['args']
		# TODO: Handle code for replaceing new paramters with older ones
		# Not handling fixed embedding layer
		model = DialogueActClassifier(args)
		model.network.load_state_dict(state_dict, strict=False)
		model.set_vocabulary(word_dict)
		return model

	@staticmethod
	def add_args(parser):
		pass


#########################################
############### NETWORK #################
#########################################
@RegisterModel('da_classifier_network')
class DialogueActClassifierNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueActClassifierNetwork, self).__init__()
		self.dialogue_embedder = DialogueEmbedder(args)

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
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		# Rejoin both directions
		conversation_encoded = conversation_encoded.view(conversation_encoded.shape[0], 1, -1).squeeze(1)

		# Reassemble the conversations
		conversation_encoded_reassembled = conversation_encoded.view(conversation_batch_size,
																max_num_utterances_batch, conversation_encoded.shape[1])
		gold_labels_reassembled = gold_labels.view(conversation_batch_size, max_num_utterances_batch)

		# Mask (Only consider the dialogues)
		dialogue_act_mask = conversation_mask #[:, 2:].contiguous()
		conversation_encoded_current = conversation_encoded_reassembled #[:, 1:-1, :].contiguous()
		gold_labels_current = gold_labels_reassembled #[:, 1:-1].contiguous()


		# Reflatten the choices, encodings, mask
		conversation_encoded_flattened = conversation_encoded_current.view(
			conversation_encoded_current.shape[0]*conversation_encoded_current.shape[1], -1)
		labels_flattened = gold_labels_current.view(gold_labels_current.shape[0]*gold_labels_current.shape[1], -1)
		mask_flattened = dialogue_act_mask.view(dialogue_act_mask.shape[0]*dialogue_act_mask.shape[1], -1)


		label_logits = self.classifier(conversation_encoded_flattened)
		new_batch_size = dialogue_act_mask.shape[0]
		new_conversation_size = dialogue_act_mask.shape[1]

		if self.args.objective == "linear_crf":
			loss = self.linear_crf(label_logits, labels_flattened, mask_flattened, new_batch_size, new_conversation_size)
		else:
			loss = self.label_cross_entropy(label_logits, labels_flattened, mask_flattened)


		return loss

	def evaluate(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch] = input

		conversation_encoded, utterance_encodings = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		# Rejoin both directions
		conversation_encoded = conversation_encoded.view(conversation_encoded.shape[0], 1, -1).squeeze(1)

		# Reassemble the conversations
		conversation_encoded_reassembled = conversation_encoded.view(conversation_batch_size,
																	 max_num_utterances_batch,
																	 conversation_encoded.shape[1])

		# Mask (Only consider the dialogues)
		dialogue_act_mask = conversation_mask #[:, 2:].contiguous()
		conversation_encoded_current = conversation_encoded_reassembled #[:, 1:-1, :].contiguous()

		# Reflatten the choices, encodings, mask
		conversation_encoded_flattened = conversation_encoded_current.view(
			conversation_encoded_current.shape[0] * conversation_encoded_current.shape[1], -1)
		mask_flattened = dialogue_act_mask.view(dialogue_act_mask.shape[0] * dialogue_act_mask.shape[1], -1)

		label_logits = self.classifier(conversation_encoded_flattened)

		## when doing CRF inference; do viterbi decoding
		h, w = conversation_encoded_current.shape[0], conversation_encoded_current.shape[1]
		if self.args.objective == "linear_crf":
			best_paths = self.structured_layer._viterbi_decode(label_logits.view(h, w, -1), dialogue_act_mask)
			best_path_ids = [b[0] for b in best_paths]
			return best_path_ids
		return label_logits

	@staticmethod
	def add_args(parser):
		model_parameters = parser.add_argument_group("Model Parameters")
		model_parameters.add_argument("--output-size", type=int)
		model_parameters.add_argument("--output-hidden-size", type=int)
		model_parameters.add_argument("--output-input-size", type=int)
		model_parameters.add_argument("--output-num-layers", type=int)


#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('da_classifier')
class DialogueActClassifier(AbstractModel):
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
		args = saved_params['args']
		# TODO: Handle code for replaceing new paramters with older ones
		# Not handling fixed embedding layer
		model = DialogueActClassifier(args)
		model.network.load_state_dict(state_dict, strict=False)
		model.set_vocabulary(word_dict)
		return model

	@staticmethod
	def add_args(parser):
		pass