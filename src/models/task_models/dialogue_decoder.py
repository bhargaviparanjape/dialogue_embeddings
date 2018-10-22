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
@RegisterModel('dl_decoder_network')
class DialogueBowNetwork(nn.Module):
	def __init__(self, args):
		super(DialogueBowNetwork, self).__init__()
		self.dialogue_embedder = DialogueEmbedder(args)
		self.args = args

		## Define class network
		dict_ = {"input_size": args.output_input_size, "hidden_size": args.output_hidden_size[0], "num_layers" : args.output_num_layers[0],
				 "output_size": args.output_size}
		self.next_utterance_decoder = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)
		self.prev_utterance_decoder = model_factory.get_model_by_name(args.output_layer[0], args, kwargs = dict_)

	def masked_loglikelihood(self, scores, target, mask):
		pass

	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask,
		max_num_utterances_batch, max_utterance_length,
		decoder_input,
		next_utterance_mask, prev_utterance_mask,
		next_utterance_word_ids, prev_utterance_word_ids,
		next_utterance_embeddings, prev_utterance_embeddings] = input

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		## Get Decoder Scores (Batch size * Max Sequence Length * Vocab size) [this lookup for start varialbe can only be done for Glove presently]
		next_utterance_embeddings = decoder_input
		next_vocabulary_scores = self.next_utterance_decoder(decoder_input, next_utterance_embeddings, conversation_encoded.squeeze(1), next_utterance_mask)
		prev_vocab_scores = self.prev_utterance_decoder(prev_utterance_embeddings, conversation_encoded.squeeze(1), prev_utterance_mask)

		## Computing custom masked cross entropy
		next_loss = self.masked_loglikelihood(next_vocabulary_scores, next_utterance_word_ids, next_utterance_mask)
		prev_loss = self.masked_loglikelihood(prev_vocab_scores, prev_utterance_word_ids, prev_utterance_mask)

		## Average loss for next and previous conversations
		loss = (next_loss + prev_loss) / 2

		return loss

	def evaluate(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, max_utterance_length] = input

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		## TODO: Get BOW Score (TBD : BEAM DECODING)
		next_vocab_scores = self.next_bow_scorer(conversation_encoded.squeeze(1))
		prev_vocab_scores = self.prev_bow_scorer(conversation_encoded.squeeze(1))

		next_predictions = torch.sort(next_vocab_scores, descending=True)[1][:, :max_utterance_length]
		prev_predictions = torch.sort(prev_vocab_scores, descending=True)[1][:, :max_utterance_length]


		return next_predictions, prev_predictions



#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('dl_decoder')
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
			input_mask = inputs[1].data.cpu()
		else:
			scores_next = scores_next.data
			scores_prev = scores_prev.data
			input_mask = inputs[1].data

		# Mask inputs
		return [scores_next, scores_prev], input_mask


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
		next_predicted = predicted[0]*mask.long()
		prev_predicted = predicted[1]*mask.long()
		next_correct = target[0]*mask.long()
		prev_correct = target[1]*mask.long()
		total = mask.sum().data.numpy() + mask.sum().data.numpy()
		correct = 0
		for i in range(predicted[0].shape[0]):
			predicted_set = set([j for j in next_predicted.numpy()[i].tolist() if j != 0])
			gold_set = set([j for j in next_correct.numpy()[i].tolist() if j != 0])
			correct += len(predicted_set & gold_set)
			predicted_set = set([j for j in prev_predicted.numpy()[i].tolist() if j != 0])
			gold_set = set([j for j in prev_correct.numpy()[i].tolist() if j != 0])
			correct += len(predicted_set & gold_set)
		metric_update_dict = {}
		metric_update_dict[self.args.metric[0]] = [correct, total]
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
		# TODO: Batch has dummy utternances that need to be specifically handled incase of average elmo
		token_embeddings, token_mask = self.token_encoder.lookup(batch)
		if self.args.use_cuda:
			token_embeddings = token_embeddings.cuda()
		input_mask_variable = variable(token_mask)

		conversation_lengths = batch['conversation_lengths']
		conversation_mask = variable(FloatTensor(batch['conversation_mask']))

		## For decoder prepare initial state
		conversation_ids = batch['utterance_word_ids']
		start_token = self.vocabulary.get_tokens(conversation_ids[0])

		## Prepare Output (If exists)
		gold_next_token_embeddings, gold_next_utterance_mask = self.token_encoder.lookup_by_name(batch,
										name_embed="next_utterance_ids", name_mask="next_utterance_mask")
		gold_next_token_ids = batch["next_utterance_ids"]
		gold_prev_token_embeddings, gold_prev_utterance_mask = self.token_encoder.lookup_by_name(batch,
										name_embed="prev_utterance_ids", name_mask="prev_utterance_mask")
		gold_prev_token_ids = batch["prev_utterance_ids"]

		# Max utterance length will be the same for next and previous utterance lists as well
		# Needs access to the token encoder itself
		if mode == "train":
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, \
				   max_num_utterances_batch, max_utterance_length, \
				   self.token_encoder, \
				   gold_next_utterance_mask, gold_prev_utterance_mask, \
				   gold_next_token_ids, gold_prev_token_ids, \
				   gold_next_token_embeddings, gold_prev_token_embeddings
		else:
			return batch_size, token_embeddings, input_mask_variable, conversation_mask, \
				   max_num_utterances_batch, max_utterance_length, \
				   self.token_encoder

	@staticmethod
	def add_args(parser):
		pass

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