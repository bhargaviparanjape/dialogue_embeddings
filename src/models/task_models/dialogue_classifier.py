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
@RegisterModel('dl_classifier_network1')
class DialogueClassifierNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueClassifierNetwork, self).__init__()
		self.dialogue_embedder = DialogueEmbedder(args)

		## Define class networkict_
		dict_ = {"input_size": args.output_input_size, "hidden_size": args.output_hidden_size, "output_size": 1,
				 "num_layers": args.output_num_layers[0], }
		self.next_dl_classifier = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)
		self.prev_dl_classifier = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)

		## Define loss function: Custom masked entropy


	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch , options_tensor,
		 goldids_next_variable, goldids_prev_variable] = input

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		## For Classification, expand each utterance by options
		## TODO: Efficient Implementation
		options = torch.index_select(conversation_encoded.squeeze(1), 0, options_tensor.view(-1))
		encoded_expand = conversation_encoded.expand(conversation_encoded.shape[0], options_tensor.shape[1],
													 conversation_encoded.shape[2]).contiguous()
		encoded_expand = encoded_expand.view(encoded_expand.shape[0] * options_tensor.shape[1], encoded_expand.shape[2])

		## Time Constrained but less memory


		## Output Layer
		next_logits = self.next_dl_classifier(encoded_expand, options)
		prev_logits = self.prev_dl_classifier(encoded_expand, options)

		## Computing custom masked cross entropy
		next_logits_flat = next_logits.view(conversation_encoded.shape[0], options_tensor.shape[1], -1)
		next_log_probs_flat = F.log_softmax(next_logits_flat, dim=1)
		prev_logits_flat = prev_logits.view(conversation_encoded.shape[0], options_tensor.shape[1], -1)
		prev_log_probs_flat = F.log_softmax(prev_logits_flat, dim=1)
		losses_flat = -torch.gather(next_log_probs_flat.squeeze(2), dim=1, index=goldids_next_variable.view(-1, 1)) \
					  + (-torch.gather(prev_log_probs_flat.squeeze(2), dim=1, index=goldids_prev_variable.view(-1, 1)))
		losses = losses_flat * conversation_mask.view(conversation_batch_size * max_num_utterances_batch, -1)

		## Average loss for next and previous conversations
		loss = losses.sum() / (2 * conversation_mask.float().sum())

		return loss

	def evaluate(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, options_tensor] = input
		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		## For Classification, expand each utterance by options
		options = torch.index_select(conversation_encoded.squeeze(1), 0, options_tensor.view(-1))
		encoded_expand = conversation_encoded.expand(conversation_encoded.shape[0], options_tensor.shape[1],
													 conversation_encoded.shape[2]).contiguous()
		encoded_expand = encoded_expand.view(encoded_expand.shape[0] * options_tensor.shape[1], encoded_expand.shape[2])

		## Output Layer
		next_logits = self.next_dl_classifier(encoded_expand, options)
		prev_logits = self.prev_dl_classifier(encoded_expand, options)

		## Maximum Values
		next_logits_flat = next_logits.view(conversation_encoded.shape[0], options_tensor.shape[1], -1)
		prev_logits_flat = prev_logits.view(conversation_encoded.shape[0], options_tensor.shape[1], -1)
		next_predictions = torch.sort(next_logits_flat.squeeze(2), descending=True)[1][:, 0]
		prev_predictions = torch.sort(prev_logits_flat.squeeze(2), descending=True)[1][:, 0]

		return next_predictions, prev_predictions



#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('dl_classifier1')
class DialogueClassifier1(AbstractModel):
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
		"""Forward a batch of examples; step the optimizer to update weights."""
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
		scores_next, scores_prev = self.network.evaluate(*inputs)

		# Convert to CPU
		if self.args.use_cuda:
			scores_next = scores_next.data.cpu()
			scores_prev = scores_prev.data.cpu()
			input_mask = inputs[2].data.cpu()
		else:
			scores_next = scores_next.data
			scores_prev = scores_prev.data
			input_mask = inputs[2].data

		# Mask inputs
		return [scores_next, scores_prev], input_mask


	def target(self, inputs):
		batch_size, *inputs = self.vectorize(inputs, mode="train")
		# Convert to CPU
		if self.args.use_cuda:
			true_next = inputs[-2].data.cpu()
			true_prev = inputs[-1].data.cpu()
			input_mask = inputs[2].data.cpu()
		else:
			true_next = inputs[-2].data
			true_prev = inputs[-1].data
			input_mask = inputs[2].data
		return [true_next, true_prev], input_mask

	def evaluate_metrics(self, predicted, target, mask, mode = "dev"):
		# Named Metric List
		mask = mask.view(-1, 1).squeeze(1)
		next_predicted = predicted[0]
		prev_predicted = predicted[1]
		next_correct = target[0]
		prev_correct = target[1]
		next_predictions_binary = (next_predicted == next_correct)
		prev_predictions_binary = (prev_predicted == prev_correct)
		prev_predictions_binary[prev_predictions_binary == 0] = 2
		correct = ((next_predictions_binary == prev_predictions_binary).long()*mask.long()).sum().numpy()
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
		## TODO: Eliminate options tensor to make faster
		options_tensor = LongTensor(batch['utterance_options_list'])
		goldids_next_variable = LongTensor(batch['next_utterance_gold'])
		goldids_prev_variable = LongTensor(batch['prev_utterance_gold'])

		if mode == "train":
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
				options_tensor, goldids_next_variable, goldids_prev_variable
		else:
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
				   options_tensor


	def init_optimizer(self):
		parameters = [p for p in self.network.parameters() if p.requires_grad]
		self.optimizer = select_optimizer(self.args, parameters)


	def parallelize(self):
		"""Use data parallel to copy the model across several gpus.
		This will take all gpus visible with CUDA_VISIBLE_DEVICES.
		"""
		self.parallel = True
		self.network = torch.nn.DataParallel(self.network)


	@staticmethod
	def add_args(parser):
		network_parameters = parser.add_argument_group("Dialogue Classifier Parameters")
		network_parameters.add_argument("--K", type=int, help="Number of options provided to dialogue classifier", default=4)


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

		# Not handling fixed embedding layer
		model = DialogueClassifier(args)
		model.network.load_state_dict(state_dict)
		model.set_vocabulary(word_dict)
		return model

#########################################
############### NETWORK #################
#########################################
@RegisterModel('dl_classifier_network')
class DialogueClassifierNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueClassifierNetwork, self).__init__()
		self.args = args
		self.dialogue_embedder = DialogueEmbedder(args)

		## Define class networkict_
		dict_ = {"input_size": args.output_input_size, "hidden_size": args.output_hidden_size, "output_size": 1,
				 "num_layers": args.output_num_layers[0], }
		self.current_dl_trasnformer1 = model_factory.get_model_by_name(args.output_layer[0], args, kwargs=dict_)
		self.current_dl_trasnformer2 = model_factory.get_model_by_name(args.output_layer[0], args, kwargs=dict_)
		dict_ = {"input_size": args.embed_size, "hidden_size": args.output_hidden_size, "output_size": 1,
				 "num_layers": args.output_num_layers[0], }
		self.next_dl_trasnformer = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)
		self.prev_dl_trasnformer = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)


	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch] = input
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		if self.args.fixed_utterance_encoder:
			utterance_encodings = token_embeddings
		else:
			utterance_encodings = self.dialogue_embedder.utterance_encoder(token_embeddings, input_mask_variable)
		utterance_encodings = utterance_encodings.view(conversation_batch_size, max_num_utterances_batch, utterance_encodings.shape[1])
		utterance_encodings_next = utterance_encodings[:, 1:, :].contiguous()
		utterance_encodings_prev = utterance_encodings[:, 0:-1, :].contiguous()

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])

		conversation_encoded_forward = conversation_encoded[:,0,:]
		conversation_encoded_backward = conversation_encoded[:, 0, :]

		conversation_encoded_forward_reassembled = conversation_encoded_forward.view(conversation_batch_size,
														max_num_utterances_batch, conversation_encoded.shape[2])
		conversation_encoded_backward_reassembled = conversation_encoded_backward.view(conversation_batch_size,
																					 max_num_utterances_batch,
																					 conversation_encoded.shape[2])

		# Shift to prepare next and previous utterence encodings
		conversation_encoded_current1 = conversation_encoded_forward_reassembled[:, 0:-1, :].contiguous()
		conversation_encoded_next = conversation_encoded_forward_reassembled[:, 1:, :].contiguous()
		conversation_mask_next = conversation_mask[:, 1:].contiguous()

		conversation_encoded_current2 = conversation_encoded_backward_reassembled[:, 1:, :].contiguous()
		conversation_encoded_previous = conversation_encoded_backward_reassembled[:, 0:-1, :].contiguous()
		# conversation_mask_previous = conversation_mask[:, 0:-1].contiguous().contiguous()

		# Gold Labels
		gold_indices = variable(LongTensor(range(conversation_encoded_current1.shape[1]))).view(-1, 1).repeat(conversation_batch_size, 1)

		# Linear transformation of both utterance representations
		transformed_current1 = self.current_dl_trasnformer1(conversation_encoded_current1)
		transformed_current2 = self.current_dl_trasnformer2(conversation_encoded_current2)


		# transformed_next = self.next_dl_trasnformer(conversation_encoded_next)
		# transformed_prev = self.prev_dl_trasnformer(conversation_encoded_previous)
		transformed_next = self.next_dl_trasnformer(utterance_encodings_next)
		transformed_prev = self.prev_dl_trasnformer(utterance_encodings_prev)

		# Output layer: Generate Scores for next and prev utterances
		next_logits = torch.bmm(transformed_current1, transformed_next.transpose(2, 1))
		prev_logits = torch.bmm(transformed_current2, transformed_prev.transpose(2, 1))

		# Computing custom masked cross entropy
		next_log_probs = F.log_softmax(next_logits, dim=2)
		prev_log_probs = F.log_softmax(prev_logits, dim=2)

		losses_next = -torch.gather(next_log_probs.view(next_log_probs.shape[0]*next_log_probs.shape[1], -1), dim=1, index=gold_indices)
		losses_prev = -torch.gather(prev_log_probs.view(prev_log_probs.shape[0]*prev_log_probs.shape[1], -1), dim=1, index=gold_indices)

		losses_masked = (losses_next.squeeze(1) * conversation_mask_next.view(conversation_mask_next.shape[0]*conversation_mask_next.shape[1]))\
						+ (losses_prev.squeeze(1) * conversation_mask_next.view(conversation_mask_next.shape[0]*conversation_mask_next.shape[1]))

		loss = losses_masked.sum() / (2*conversation_mask_next.float().sum())

		return loss

	def evaluate(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch] = input

		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		if self.args.fixed_utterance_encoder:
			utterance_encodings = token_embeddings
		else:
			utterance_encodings = self.dialogue_embedder.utterance_encoder(token_embeddings, input_mask_variable)
		utterance_encodings = utterance_encodings.view(conversation_batch_size, max_num_utterances_batch, utterance_encodings.shape[1])
		utterance_encodings_next = utterance_encodings[:, 1:, :].contiguous()
		utterance_encodings_prev = utterance_encodings[:, 0:-1, :].contiguous()

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])

		conversation_encoded_forward = conversation_encoded[:, 0, :]
		conversation_encoded_backward = conversation_encoded[:, 0, :]

		conversation_encoded_forward_reassembled = conversation_encoded_forward.view(conversation_batch_size,
																					 max_num_utterances_batch,
																					 conversation_encoded.shape[2])
		conversation_encoded_backward_reassembled = conversation_encoded_backward.view(conversation_batch_size,
																					   max_num_utterances_batch,
																					   conversation_encoded.shape[2])

		# Shift to prepare next and previous utterence encodings
		conversation_encoded_current1 = conversation_encoded_forward_reassembled[:, 0:-1, :]
		conversation_encoded_next = conversation_encoded_forward_reassembled[:, 1:, :]
		conversation_mask_next = conversation_mask[:, 1:].contiguous()

		conversation_encoded_current2 = conversation_encoded_backward_reassembled[:, 1:, :]
		conversation_encoded_previous = conversation_encoded_backward_reassembled[:, 0:-1, :]
		conversation_mask_previous = conversation_mask[:, 0:-1].contiguous()

		# Linear transformation of both utterance representations
		transformed_current1 = self.current_dl_trasnformer1(conversation_encoded_current1)
		transformed_current2 = self.current_dl_trasnformer2(conversation_encoded_current2)

		# transformed_next = self.next_dl_trasnformer(conversation_encoded_next)
		# transformed_prev = self.prev_dl_trasnformer(conversation_encoded_previous)
		transformed_next = self.next_dl_trasnformer(utterance_encodings_next)
		transformed_prev = self.prev_dl_trasnformer(utterance_encodings_prev)

		# Output layer: Generate Scores for next and prev utterances
		next_logits = torch.bmm(transformed_current1, transformed_next.transpose(2, 1))
		prev_logits = torch.bmm(transformed_current2, transformed_prev.transpose(2, 1))

		next_predictions = torch.sort(next_logits, dim=2, descending=True)[1][:, 0]
		prev_predictions = torch.sort(prev_logits, dim=2, descending=True)[1][:, 0]

		return next_predictions, prev_predictions

#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('dl_classifier')
class DialogueClassifier(AbstractModel):
	def __init__(self, args):

		## Initialize environment
		self.args = args
		self.updates = 0

		## If token encodings are not computed on the fly using character CNN based models but are obtained from a pretrained model
		if args.fixed_token_encoder:
			self.token_encoder = model_factory.get_embeddings(args.token_encoder, args)
		self.network = model_factory.get_model_by_name(args.network, args)

	def vectorize(self, batch, mode="train"):
		batch_size = int(len(batch['utterance_list']) / batch['max_num_utterances'])
		max_num_utterances_batch = batch['max_num_utterances']

		# TODO: Batch has dummy utternances that need to be specifically handled incase of average elmo
		token_embeddings, token_mask = self.token_encoder.lookup(batch)

		if self.args.use_cuda:
			token_embeddings = token_embeddings.cuda()
		input_mask_variable = variable(token_mask)

		conversation_lengths = batch['conversation_lengths']
		conversation_mask = variable(FloatTensor(batch['conversation_mask']))

		if mode == "train":
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch
		else:
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch

	def predict(self, inputs):
		# Eval mode
		self.network.eval()

		# Run forward
		batch_size, *inputs = self.vectorize(inputs, mode = "test")
		scores_next, scores_prev = self.network.evaluate(*inputs)

		# Convert to CPU
		if self.args.use_cuda:
			scores_next = scores_next.data.cpu()
			scores_prev = scores_prev.data.cpu()
			input_mask = inputs[2].data.cpu()
		else:
			scores_next = scores_next.data
			scores_prev = scores_prev.data
			input_mask = inputs[2].data

		# Mask inputs
		return [scores_next, scores_prev], input_mask


	def target(self, inputs):
		batch_size, *inputs = self.vectorize(inputs, mode="train")
		# Convert to CPU
		if self.args.use_cuda:
			input_mask = inputs[2].data.cpu()
		else:
			input_mask = inputs[2].data
		return None, input_mask

	def evaluate_metrics(self, predicted, target, mask, mode = "dev"):
		# Named Metric List
		batch_size = mask.shape[0]
		next_mask = mask[:, 0:-1].contiguous().view(-1,1).long()
		next_predicted = predicted[0].contiguous().view(-1,1)
		prev_predicted = predicted[1].contiguous().view(-1,1)
		correct = torch.LongTensor(range(predicted[0].shape[1])).view(-1, 1).repeat(batch_size, 1)
		next_predictions_binary = (next_predicted == correct).long()*next_mask
		prev_predictions_binary = (prev_predicted == correct).long()*next_mask
		prev_predictions_binary[prev_predictions_binary == 0] = 2
		correct = ((next_predictions_binary == prev_predictions_binary).long()*next_mask.long()).sum().numpy()
		total = next_mask.sum().data.numpy()
		metric_update_dict = {}
		metric_update_dict[self.args.metric[0]] = [correct, total]
		return metric_update_dict

	def set_vocabulary(self, vocabulary):
		self.vocabulary = vocabulary
		## Embedding layer initialization depends upon vocabulary
		if hasattr(self.token_encoder, "load_embeddings"):
			self.token_encoder.load_embeddings(self.vocabulary.vocabulary)

	@staticmethod
	def add_args(parser):
		pass