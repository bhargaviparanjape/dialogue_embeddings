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
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

#########################################
############### NETWORK #################
#########################################
@RegisterModel('dl_bow_network1')
class DialogueBowNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueBowNetwork, self).__init__()
		self.dialogue_embedder = DialogueEmbedder(args)
		self.args = args

		## Define class network
		dict_ = {"input_size": args.output_input_size, "hidden_size": args.output_hidden_size[0], "num_layers" : args.output_num_layers[0],
				 "output_size": args.output_size}
		self.next_bow_scorer = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)
		self.prev_bow_scorer = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)

		## Define loss function: Custom masked entropy


	def multilabel_cross_entropy(self, input, target, mask):
		negative_log_prob = -(F.log_softmax(input/self.args.temperature))
		#TODO: Divide by length of each utterance and divide by batch size
		loss = (torch.gather(negative_log_prob, 1, target) * mask.float()).sum()/mask.float().sum()
		# loss = (negative_log_prob*target.float()).sum()/target.float().sum()
		return loss

	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch,
		gold_next_mask, gold_prev_mask, gold_next_bow, gold_prev_bow] = input

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		## Get BOW Score
		next_vocab_scores = self.next_bow_scorer(conversation_encoded.squeeze(1))
		prev_vocab_scores = self.prev_bow_scorer(conversation_encoded.squeeze(1))

		## Computing custom masked cross entropy
		next_loss = self.multilabel_cross_entropy(next_vocab_scores, gold_next_bow, gold_next_mask)
		prev_loss = self.multilabel_cross_entropy(prev_vocab_scores, gold_prev_bow, gold_prev_mask)

		## Average loss for next and previous conversations
		loss = (next_loss + prev_loss) / 2
		#loss = next_loss


		return loss

	def evaluate(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch] = input

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		## Get BOW Score
		next_vocab_scores = self.next_bow_scorer(conversation_encoded.squeeze(1))
		prev_vocab_scores = self.prev_bow_scorer(conversation_encoded.squeeze(1))

		next_vocab_probabilities = F.softmax(next_vocab_scores, dim=1)
		prev_vocab_probabilities = F.softmax(prev_vocab_scores, dim=1)

		## Maximum Values above a threshold hyperparameter
		## Loop over batch (??)
		next_predictions = torch.sort(next_vocab_probabilities, descending=True)[0][:,]
		prev_predictions = torch.sort(prev_vocab_probabilities, descending=True)[0][:,]

		return next_vocab_probabilities, prev_vocab_probabilities
		# return next_vocab_probabilities


#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('dl_bow1')
class DialogueClassifier(AbstractModel):
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
		# scores_next = self.network.evaluate(*inputs)

		# Convert to CPU
		if self.args.use_cuda:
			scores_next = scores_next.data.cpu()
			scores_prev = scores_prev.data.cpu()
			input_mask = inputs[1].data.cpu()
		else:
			scores_next = scores_next.data
			scores_prev = scores_prev.data
			input_mask = inputs[1].data

		# Mask inputs
		return [scores_next, scores_prev], input_mask
		# return [scores_next], input_mask


	def target(self, inputs):
		batch_size, *inputs = self.vectorize(inputs, mode="train")
		# Convert to CPU
		if self.args.use_cuda:
			true_next = inputs[-2].data.cpu()
			true_prev = inputs[-1].data.cpu()
			input_mask = inputs[1].data.cpu()
		else:
			true_next = inputs[-2].data
			true_prev = inputs[-1].data
			input_mask = inputs[1].data
		return [true_next, true_prev], input_mask

	def evaluate_metrics(self, predicted, target, mask, mode = "dev"):
		# Named Metric List
		mask = mask.view(-1, 1).squeeze(1)
		next_predicted = predicted[0].numpy()
		prev_predicted = predicted[1].numpy()
		next_correct = target[0].numpy()
		prev_correct = target[1].numpy()
		batch_precision = 0
		batch_recall = 0
		batch_f1 = 0
		total = 0
		# TODO: Replace by confusion matrix + F1 from sklearn to get all metrics
		for i in range(next_predicted.shape[0]):
			predicted_ids = np.where(next_predicted[i] > self.args.threshold)[0] # remove start, end, pad token
			gold_ids = next_correct[i][np.where(next_correct[i] != 0)]
			if len(gold_ids) == 0:
				continue

			if len(set(predicted_ids)) == 0:
				precision = 0
			else:
				precision = float(len(set(gold_ids)&set(predicted_ids)))/len(set(predicted_ids))
			recall = float(len(set(gold_ids) & set(predicted_ids))) / len(set(gold_ids))
			if precision + recall == 0:
				f1 = 0
			else:
				f1 = (2*precision*recall)/(precision + recall)
			batch_precision += precision
			batch_recall += recall
			batch_f1 += f1

			predicted_ids = np.where(prev_predicted[i] > self.args.threshold)[0]
			gold_ids = prev_correct[0][np.where(prev_correct[0] != 0)]

			if len(set(predicted_ids)) == 0:
				precision = 0
			else:
				precision = float(len(set(gold_ids) & set(predicted_ids))) / len(set(predicted_ids))
			recall = float(len(set(gold_ids) & set(predicted_ids))) / len(set(gold_ids))
			if precision + recall == 0:
				f1 = 0
			else:
				f1 = (2 * precision * recall) / (precision + recall)
			batch_precision += precision
			batch_recall += recall
			batch_f1 += f1

			total += 1

		metric_update_dict = {}

		metric_update_dict["precision"] = [batch_precision, 2*total]
		metric_update_dict["recall"] = [batch_recall, 2 * total]
		metric_update_dict["f1"] = [batch_f1, 2 * total]
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
		max_utterance_length = batch['max_utterance_length']

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
		gold_next_bow_vectors = LongTensor(batch['next_bow_list'])
		gold_prev_bow_vectors = LongTensor(batch['prev_bow_list'])
		gold_next_bow_mask = LongTensor(batch['next_bow_mask'])
		gold_prev_bow_mask = LongTensor(batch['prev_bow_mask'])

		if mode == "train":
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
				gold_next_bow_mask, gold_prev_bow_mask, gold_next_bow_vectors, gold_prev_bow_vectors
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


	@staticmethod
	def add_args(parser):
		pass


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
@RegisterModel('dl_bow_network')
class DialogueBowNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueBowNetwork, self).__init__()
		self.dialogue_embedder = DialogueEmbedder(args)
		self.args = args

		## Define class network
		dict_ = {"input_size": args.output_input_size + args.embed_size, "hidden_size": args.output_hidden_size, "num_layers" : args.output_num_layers,
				 "output_size": args.output_size}
		self.next_bow_scorer = model_factory.get_model_by_name(args.output_layer, args, kwargs = dict_)
		self.prev_bow_scorer = model_factory.get_model_by_name(args.output_layer, args, kwargs = dict_)




	def multilabel_cross_entropy(self, input, target, mask):
		negative_log_prob = -(F.log_softmax(input/self.args.temperature, dim=1))
		#TODO: Divide by length of each utterance and divide by batch size
		loss = (torch.gather(negative_log_prob, 1, target) * mask.float()).sum()/mask.float().sum()
		# loss = (negative_log_prob*target.float()).sum()/target.float().sum()
		return loss

	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch,
		gold_bow, gold_mask] = input

		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)
		conversation_encoded, utterance_encodings = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])

		utterance_encodings = utterance_encodings.view(conversation_batch_size, max_num_utterances_batch,
		                                               utterance_encodings.shape[1])
		utterance_encodings_next = utterance_encodings[:, 0:-1, :].contiguous().view(-1, utterance_encodings.shape[-1])
		utterance_encodings_prev = utterance_encodings[:, 1:, :].contiguous().view(-1, utterance_encodings.shape[-1])

		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)
		conversation_encoded_forward = conversation_encoded[:,0,:]
		conversation_encoded_backward = conversation_encoded[:, 1, :]

		# Reassemble into conversations
		conversation_encoded_forward_reassembled = conversation_encoded_forward.view(conversation_batch_size,
																					 max_num_utterances_batch,
																					 conversation_encoded_forward.shape[
																						 1])
		conversation_encoded_backward_reassembled = conversation_encoded_backward.view(conversation_batch_size,
																					   max_num_utterances_batch,
																					   conversation_encoded_backward.shape[
																						   1])
		bow_reassembled = gold_bow.view(conversation_batch_size, max_num_utterances_batch, gold_bow.shape[1])
		bow_mask_reassembled = gold_mask.view(conversation_batch_size, max_num_utterances_batch, gold_mask.shape[1])

		# Shift to prepare next and previous utterence encodings
		conversation_encoded_current1 = conversation_encoded_forward_reassembled[:, 0:-1, :].contiguous()
		bow_next = bow_reassembled[:, 1:, :].contiguous()
		conversation_mask_next = bow_mask_reassembled[:, 1:].contiguous()

		conversation_encoded_current2 = conversation_encoded_backward_reassembled[:, 1:, :].contiguous()
		bow_previous = bow_reassembled[:, 0:-1, :].contiguous()
		conversation_mask_previous = bow_mask_reassembled[:, 0:-1].contiguous()

		# Flatten masks and inputs
		conversation_encoded_current1 = conversation_encoded_current1.view(conversation_encoded_current1.shape[0]*
																		   conversation_encoded_current1.shape[1], -1)
		conversation_encoded_current2 = conversation_encoded_current2.view(conversation_encoded_current2.shape[0]*
																		   conversation_encoded_current2.shape[1], -1)
		conversation_mask_next = conversation_mask_next.view(conversation_mask_next.shape[0]*
																		   conversation_mask_next.shape[1], -1)
		conversation_mask_previous = conversation_mask_previous.view(conversation_mask_previous.shape[0]*
																		   conversation_mask_previous.shape[1], -1)
		bow_next = bow_next.view(bow_next.shape[0]*bow_next.shape[1], -1)
		bow_previous = bow_previous.view(bow_previous.shape[0]*bow_previous.shape[1], -1)

		## Get BOW Score
		## Utterrence scores are also joint in for propagation
		next_vocab_scores = self.next_bow_scorer(torch.cat((conversation_encoded_current1, utterance_encodings_next), 1))
		prev_vocab_scores = self.prev_bow_scorer(torch.cat((conversation_encoded_current2, utterance_encodings_prev), 1))

		## Computing custom masked cross entropy
		next_loss = self.multilabel_cross_entropy(next_vocab_scores, bow_next, conversation_mask_next)
		prev_loss = self.multilabel_cross_entropy(prev_vocab_scores, bow_previous, conversation_mask_next)

		## Average loss for next and previous conversations
		loss = (next_loss + prev_loss) / 2

		return loss

	def evaluate(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch] = input

		conversation_encoded, utterance_encodings = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])

		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)
		conversation_encoded_forward = conversation_encoded[:, 0, :]
		conversation_encoded_backward = conversation_encoded[:, 1, :]

		utterance_encodings = utterance_encodings.view(conversation_batch_size, max_num_utterances_batch,
		                                               utterance_encodings.shape[1])
		utterance_encodings_next = utterance_encodings[:, 0:-1, :].contiguous().view(-1, utterance_encodings.shape[-1])
		utterance_encodings_prev = utterance_encodings[:, 1:, :].contiguous().view(-1, utterance_encodings.shape[-1])

		# Reassemble into conversations
		conversation_encoded_forward_reassembled = conversation_encoded_forward.view(conversation_batch_size,
																					 max_num_utterances_batch,
																					 conversation_encoded_forward.shape[
																						 1])
		conversation_encoded_backward_reassembled = conversation_encoded_backward.view(conversation_batch_size,
																					   max_num_utterances_batch,
																					   conversation_encoded_backward.shape[
																						   1])

		# Shift to prepare next and previous utterence encodings
		conversation_encoded_current1 = conversation_encoded_forward_reassembled[:, 0:-1, :].contiguous()

		conversation_encoded_current2 = conversation_encoded_backward_reassembled[:, 1:, :].contiguous()

		# Flatten masks and inputs
		conversation_encoded_current1 = conversation_encoded_current1.view(conversation_encoded_current1.shape[0] *
																		   conversation_encoded_current1.shape[1], -1)
		conversation_encoded_current2 = conversation_encoded_current2.view(conversation_encoded_current2.shape[0] *
																		   conversation_encoded_current2.shape[1], -1)
		## Get BOW Score
		next_vocab_scores = self.next_bow_scorer(
			torch.cat((conversation_encoded_current1, utterance_encodings_next), 1))
		prev_vocab_scores = self.prev_bow_scorer(
			torch.cat((conversation_encoded_current2, utterance_encodings_prev), 1))

		next_vocab_probabilities = F.softmax(next_vocab_scores, dim=1)
		prev_vocab_probabilities = F.softmax(prev_vocab_scores, dim=1)

		return next_vocab_probabilities, prev_vocab_probabilities



#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('dl_bow')
class DialogueClassifier(AbstractModel):
	def __init__(self, args):

		## Initialize environment
		self.args = args
		self.updates = 0

		## If token encodings are not computed on the fly using character CNN based models but are obtained from a pretrained model
		if args.fixed_token_encoder:
			self.token_encoder = model_factory.get_embeddings(args.token_encoder, args)

		self.network = model_factory.get_model_by_name(args.network, args)

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
			input_mask = inputs[2].view(batch_size, -1).data.cpu()
		else:
			scores_next = scores_next.data
			scores_prev = scores_prev.data
			input_mask = inputs[2].view(batch_size, -1).data

		# Mask inputs
		return [scores_next, scores_prev], input_mask


	def target(self, inputs):
		batch_size, *inputs = self.vectorize(inputs, mode="train")
		# Convert to CPU
		if self.args.use_cuda:
			utterance_bow = inputs[-2].data.cpu()
			input_mask = inputs[2].data.cpu()
		else:
			utterance_bow = inputs[-2].data
			input_mask = inputs[2].data
		return [utterance_bow], input_mask

	def evaluate_metrics(self, predicted, target, mask, mode = "dev"):
		# Named Metric List
		next_mask = mask[:, 1:].contiguous().view(-1,1).long()
		next_predicted = predicted[0].numpy()
		prev_predicted = predicted[1].numpy()
		target_reassembled = target[0].view(mask.shape[0], -1, target[0].shape[1])
		next_correct = target_reassembled[:, 1:, :].contiguous().view(next_mask.shape[0],-1)
		prev_correct = target_reassembled[:, 0:-1, :].contiguous().view(next_mask.shape[0],-1)
		if self.args.use_cuda:
			next_correct = next_correct.cpu().data.numpy()
			prev_correct = prev_correct.cpu().data.numpy()
		else:
			next_correct = next_correct.data.numpy()
			prev_correct = prev_correct.data.numpy()
		batch_precision = 0
		batch_recall = 0
		batch_f1 = 0
		total = 0
		# TODO: Replace by confusion matrix + F1 from sklearn to get all metrics
		for i in range(next_predicted.shape[0]):
			predicted_ids = np.where(next_predicted[i] > self.args.threshold)[0] # remove start, end, pad token
			gold_ids = next_correct[i][np.where(next_correct[i] != 0)]
			if len(gold_ids) == 0:
				continue

			if len(set(predicted_ids)) == 0:
				precision = 0
			else:
				precision = float(len(set(gold_ids)&set(predicted_ids)))/len(set(predicted_ids))
			recall = float(len(set(gold_ids) & set(predicted_ids))) / len(set(gold_ids))
			if precision + recall == 0:
				f1 = 0
			else:
				f1 = (2*precision*recall)/(precision + recall)
			batch_precision += precision
			batch_recall += recall
			batch_f1 += f1

			predicted_ids = np.where(prev_predicted[i] > self.args.threshold)[0]
			gold_ids = prev_correct[0][np.where(prev_correct[0] != 0)]

			if len(set(predicted_ids)) == 0:
				precision = 0
			else:
				precision = float(len(set(gold_ids) & set(predicted_ids))) / len(set(predicted_ids))
			recall = float(len(set(gold_ids) & set(predicted_ids))) / len(set(gold_ids))
			if precision + recall == 0:
				f1 = 0
			else:
				f1 = (2 * precision * recall) / (precision + recall)
			batch_precision += precision
			batch_recall += recall
			batch_f1 += f1

			total += 1

		metric_update_dict = {}

		metric_update_dict["precision"] = [batch_precision, 2*total]
		metric_update_dict["recall"] = [batch_recall, 2 * total]
		metric_update_dict["f1"] = [batch_f1, 2 * total]
		return metric_update_dict

	def set_vocabulary(self, vocabulary):
		self.vocabulary = vocabulary
		## Embedding layer initialization depends upon vocabulary
		if hasattr(self.token_encoder, "load_embeddings"):
			self.token_encoder.load_embeddings(self.vocabulary)

	def vectorize(self, batch, mode = "train"):
		batch_size = int(len(batch['utterance_list']) / batch['max_num_utterances'])
		max_num_utterances_batch = batch['max_num_utterances']
		max_utterance_length = batch['max_utterance_length']

		## Prepare Token Embeddings
		token_embeddings, token_mask = self.token_encoder.lookup(batch)
		if self.args.use_cuda:
			token_embeddings = token_embeddings.cuda()
		input_mask_variable = variable(token_mask)

		conversation_lengths = batch['conversation_lengths']
		conversation_mask = variable(FloatTensor(batch['conversation_mask']))

		## Prepare Ouput (If exists)
		bow_list = LongTensor(batch['utterance_bow_list'])
		bow_mask = LongTensor(batch['utterance_bow_mask'])

		if mode == "train":
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
				bow_list, bow_mask
		else:
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch


	@staticmethod
	def add_args(parser):
		network_parameters = parser.add_argument_group("Dialogue Classifier Parameters")
		network_parameters.add_argument("--threshold", type=float, help="Threshold to choose words from vocabulary", default=0.001)
		network_parameters.add_argument("--temperature", type=float, help="Threshold to choose words from vocabulary",
										default=0.5)

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
