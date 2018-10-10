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
@RegisterModel('dl_bow_network')
class DialogueBowNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueBowNetwork, self).__init__()
		self.dialogue_embedder = DialogueEmbedder(args)
		self.args = args

		## Define class network
		self.next_bow_scorer = model_factory.get_model_by_name(args.output_layer[0], args)
		# self.prev_bow_scorer = model_factory.get_model_by_name(args.output_layer[0], args)

		## Define loss function: Custom masked entropy


	def multilabel_cross_entropy(self, input, target,mask):
		negative_log_prob = -(F.log_softmax(input))
		#TODO: Divide by length of each utterance and divide by batch size
		loss = (negative_log_prob*target.float()).sum()/target.float().sum()
		return loss

	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch,
		gold_next_bow, gold_prev_bow] = input

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		## Get BOW Score
		next_vocabulary_scores = self.next_bow_scorer(conversation_encoded.squeeze(1))
		#prev_vocab_scores = self.prev_bow_scorer(conversation_encoded.squeeze(1))

		## Computing custom masked cross entropy
		next_loss = self.multilabel_cross_entropy(next_vocabulary_scores, gold_next_bow, input_mask_variable)
		#prev_loss = self.multilabel_cross_entropy(prev_vocab_scores, gold_prev_bow, input_mask_variable)

		## Average loss for next and previous conversations
		# loss = (next_loss + prev_loss) / 2
		loss = next_loss


		return loss

	def evaluate(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch] = input

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		## Get BOW Score
		next_vocab_scores = self.next_bow_scorer(conversation_encoded.squeeze(1))
		# prev_vocab_scores = self.prev_bow_scorer(conversation_encoded.squeeze(1))

		next_vocab_probabilities = F.softmax(next_vocab_scores, dim=1)
		# prev_vocab_probabilities = F.softmax(prev_vocab_scores, dim=1)

		## Maximum Values above a threshold hyperparameter
		## Loop over batch (??)
		# next_predictions = torch.sort(next_vocab_probabilities, descending=True)[0][:,]
		# prev_predictions = torch.sort(prev_vocab_probabilities, descending=True)[0][:,]

		# return next_vocab_probabilities, prev_vocab_probabilities
		return next_vocab_probabilities


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
		# scores_next, scores_prev = self.network.evaluate(*inputs)
		scores_next = self.network.evaluate(*inputs)

		# Convert to CPU
		if self.args.use_cuda:
			scores_next = scores_next.data.cpu()
			# scores_prev = scores_prev.data.cpu()
			input_mask = inputs[2].data.cpu()
		else:
			scores_next = scores_next.data
			# scores_prev = scores_prev.data
			input_mask = inputs[2].data

		# Mask inputs
		# return [scores_next, scores_prev], input_mask
		return [scores_next], input_mask


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
		return [true_next], input_mask

	def evaluate_metrics(self, predicted, target, mask, mode = "dev"):
		# Named Metric List
		mask = mask.view(-1, 1).squeeze(1)
		next_predicted = predicted[0].numpy()
		#prev_predicted = predicted[1].numpy()
		next_correct = target[0].numpy()
		#prev_correct = target[1].numpy()
		correct = 0
		total = next_correct.sum() #+ prev_correct.sum()
		# TODO: Replace by confusion matrix + F1 from sklearn to get all metrics
		for i in range(next_predicted.shape[0]):
			predicted_ids = np.where(next_predicted[0] > self.args.T)[0]
			gold_ids = np.where(next_correct[0] > self.args.T)[0]
			correct += len(set(gold_ids)&set(predicted_ids))
			#predicted_ids = np.where(prev_predicted[0] > self.args.T)[0]
			#gold_ids = np.where(prev_correct[0] > self.args.T)[0]
			#correct += len(set(gold_ids) & set(predicted_ids))
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

		if mode == "train":
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
				gold_next_bow_vectors, gold_prev_bow_vectors
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
		network_parameters = parser.add_argument_group("Dialogue Classifier Parameters")
		network_parameters.add_argument("--T", type=float, help="Threshold to choose words from vocabulary", default=0.00039)


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
