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
@RegisterModel('da_bow_network')
class DialogueBowNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueBowNetwork, self).__init__()
		self.dialogue_embedder = DialogueEmbedder(args)
		self.args = args

		## Define class network
		dict_ = {"input_size": args.output_input_size, "hidden_size": args.output_hidden_size[0], "num_layers" : args.output_num_layers[0],
				 "output_size": args.output_size[0]}
		self.next_bow_scorer = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)
		self.prev_bow_scorer = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)
		dict_ = {"input_size": args.output_input_size, "hidden_size": args.output_hidden_size[1], "num_layers" : args.output_num_layers[1],
				 "output_size": args.output_size[1]}
		self.classifier = model_factory.get_model_by_name(args.output_layer[1], args, kwargs = dict_)
		## Define loss function: Custom masked entropy


	def multilabel_cross_entropy(self, input, target, mask):
		negative_log_prob = -(F.log_softmax(input/self.args.temperature))
		loss = (torch.gather(negative_log_prob, 1, target) * mask.float()).sum()/mask.float().sum()
		# loss = (negative_log_prob*target.float()).sum()/target.float().sum()
		return loss

	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch,
		gold_next_mask, gold_prev_mask, gold_next_bow, gold_prev_bow, gold_labels] = input

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

		label_logits = self.classifier(conversation_encoded.squeeze(1))
		label_log_probs_flat = F.log_softmax(label_logits, dim=1)
		label_losses_flat = -torch.gather(label_log_probs_flat, dim=1, index=gold_labels.view(-1, 1))
		label_losses = label_losses_flat * conversation_mask.view(conversation_batch_size * max_num_utterances_batch,
																  -1)
		label_loss = label_losses.sum() / conversation_mask.float().sum()

		combined_loss = self.args.output_weights[0] * loss + self.args.output_weights[1] * label_loss
		return combined_loss

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

		label_logits = self.classifier(conversation_encoded.squeeze(1))

		return next_vocab_probabilities, prev_vocab_probabilities, label_logits



#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('da_bow')
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
		scores_next, scores_prev, label_logits = self.network.evaluate(*inputs)
		labels_predictions = torch.sort(label_logits, descending=True)[1][:, 0]
		# Convert to CPU
		if self.args.use_cuda:
			scores_next = scores_next.data.cpu()
			scores_prev = scores_prev.data.cpu()
			labels_predictions = labels_predictions.data.cpu()
			input_mask = inputs[1].data.cpu()
			conversation_mask = inputs[2].data.cpu()
		else:
			scores_next = scores_next.data
			scores_prev = scores_prev.data
			labels_predictions = labels_predictions.data
			input_mask = inputs[1].data
			conversation_mask = inputs[2].data
		# Mask inputs
		return [scores_next, scores_prev, labels_predictions], [input_mask, conversation_mask]


	def target(self, inputs):
		batch_size, *inputs = self.vectorize(inputs, mode="train")
		# Convert to CPU
		if self.args.use_cuda:
			true_next = inputs[-3].data.cpu()
			true_prev = inputs[-2].data.cpu()
			true_labels = inputs[-1].data.cpu()
			input_mask = inputs[1].data.cpu()
			conversation_mask = inputs[2].data.cpu()
		else:
			true_next = inputs[-3].data
			true_prev = inputs[-2].data
			true_labels = inputs[-1].data
			input_mask = inputs[1].data
			conversation_mask = inputs[2].data
		return [true_next, true_prev, true_labels], [input_mask, conversation_mask]

	def evaluate_metrics(self, predicted, target, mask, mode = "dev"):
		# Named Metric List
		conv_mask = mask[1].view(-1, 1).squeeze(1)
		next_predicted = predicted[0].numpy()
		prev_predicted = predicted[1].numpy()
		labels_predicted = predicted[2]
		next_correct = target[0].numpy()
		prev_correct = target[1].numpy()
		labels_correct = target[2]
		batch_precision = 0
		batch_recall = 0
		batch_f1 = 0
		total = 0
		for i in range(next_predicted.shape[0]):
			predicted_ids = np.where(next_predicted[i] > self.args.threshold)[0]
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

		predictions_binary = (labels_predicted == labels_correct)
		correct_labels = (predictions_binary.long() * conv_mask.long()).sum().numpy()
		total_labels = conv_mask.sum().data.numpy()

		metric_update_dict = {}

		metric_update_dict["precision"] = [batch_precision, 2*total]
		metric_update_dict["recall"] = [batch_recall, 2 * total]
		metric_update_dict["f1"] = [batch_f1, 2 * total]
		metric_update_dict["accuracy"] = [correct_labels, total_labels]
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
		utterance_labels = LongTensor(batch['label'])

		if mode == "train":
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
				gold_next_bow_mask, gold_prev_bow_mask, gold_next_bow_vectors, gold_prev_bow_vectors, utterance_labels
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