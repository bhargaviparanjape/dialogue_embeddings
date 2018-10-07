import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import pdb

from src.models.abstract_model import AbstractModel
from src.models import factory as model_factory
from src.learn import factory as learn_factory
from src.models.factory import RegisterModel
from src.models.components.output_models.dialogue_embedder import DialogueEmbedder
from src.utils.utility_functions import variable,FloatTensor,ByteTensor,LongTensor,select_optimizer


#########################################
############### NETWORK #################
#########################################
@RegisterModel('dl_classifier_network')
class DialogueClassifierNetwork(nn.Module):
	def __init__(self, args, logger = None):
		super(DialogueClassifierNetwork, self).__init__()
		self.dialogue_embedder = DialogueEmbedder(args, logger)

		## Define class network
		self.next_dl_classifier = model_factory.get_model_by_name(args.output_layer, args)
		self.prev_dl_classifier = model_factory.get_model_by_name(args.output_layer, args)

		## Define loss function: Custom masked entropy


	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch ,
		 options_tensor, goldids_next_variable, goldids_prev_variable] = input[0]

		conversation_encoded = self.dialogue_embedder([token_embeddings, input_mask_variable, conversation_mask,
													   max_num_utterances_batch])
		conversation_batch_size = int(token_embeddings.shape[0] / max_num_utterances_batch)

		## For Classification, expand each utterance by options
		## TODO: Efficient Implementation
		options = torch.index_select(conversation_encoded.squeeze(1), 0, options_tensor.view(-1))
		encoded_expand = conversation_encoded.expand(conversation_encoded.shape[0], options_tensor.shape[1],
													 conversation_encoded.shape[2]).contiguous()
		encoded_expand = encoded_expand.view(encoded_expand.shape[0] * options_tensor.shape[1], encoded_expand.shape[2])

		## Output Layer
		next_logits = self.next_dl_classifier(encoded_expand, options)
		prev_logits = self.prev_dl_classifier(encoded_expand, options)

		## Computing custom masked cross entropy
		next_logits_flat = next_logits.view(conversation_encoded.shape[0], options_tensor.shape[1], -1)
		next_log_probs_flat = F.log_softmax(next_logits_flat)
		prev_logits_flat = prev_logits.view(conversation_encoded.shape[0], options_tensor.shape[1], -1)
		prev_log_probs_flat = F.log_softmax(prev_logits_flat)
		losses_flat = -torch.gather(next_log_probs_flat.squeeze(2), dim=1, index=goldids_next_variable.view(-1, 1)) \
					  + (-torch.gather(prev_log_probs_flat.squeeze(2), dim=1, index=goldids_prev_variable.view(-1, 1)))
		losses = losses_flat * conversation_mask.view(conversation_batch_size * max_num_utterances_batch, -1)

		## Average loss for next and previous conversations
		loss = losses.sum() / (2 * conversation_mask.float().sum())

		return loss

	def eval(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch,
		 options_tensor] = input[0]
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

		return next_logits, prev_logits



#################################################
############### NETWORK WRAPPER #################
#################################################
@RegisterModel('dl_classifier')
class DialogueClassifier(AbstractModel):
	def __init__(self, args, logger = None):

		## Initialize environment
		self.args = args
		self.logger = logger
		self.updates = 0

		## If token encodings are not computed on the fly using character CNN based models but are obtained from a pretrained model
		if args.fixed_token_encoder:
			self.token_encoder = model_factory.get_embeddings(args.pretrained_embedding_path, args, logger)

		self.network = model_factory.get_model_by_name(args.network, args, logger)

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
		return loss.data[0], batch_size


	def checkpoint(self, file_path, epoch_no):
		raise NotImplementedError

	def predict(self, inputs):
		# Eval mode
		self.network.eval()

		# Run forward
		batch_size, *inputs = self.vectorize(inputs, mode = "test")
		scores_next, scores_prev, batch_size = self.network.eval(*inputs)

		# Convert to CPU
		if self.args.use_cuda:
			scores_next = scores_next.data.cpu()
			scores_prev = scores_prev.data.cpu()
			input_mask = inputs[0][2].data.cpu()
		else:
			scores_next = scores_next.data
			scores_prev = scores_prev.data
			input_mask = inputs[0][2].data

		# Mask inputs
		return [scores_next, scores_prev], input_mask


	def target(self, inputs):
		batch_size, *inputs = self.vectorize(inputs, mode="train")
		# Convert to CPU
		if self.args.use_cuda:
			true_next = inputs[0][-2].data.cpu()
			true_prev = inputs[0][-1].data.cpu()
			input_mask = inputs[0][2].data.cpu()
		else:
			true_next = inputs[0][-2].data
			true_prev = inputs[0][-1].data
			input_mask = inputs[0][2].data
		return [true_next, true_prev], input_mask

	def evaluate_metrics(self, predicted, target, mask, mode = "dev"):
		# Named Metric List
		mask = mask.view(-1, 1).squeeze(1)
		next_predicted = predicted[0]
		prev_predicted = predicted[1]
		next_correct = target[0]
		prev_correct = target[1]
		correct = (((next_predicted == next_correct) == (prev_predicted == prev_correct)).long()*mask).sum().numpy()
		total = mask.sum().data.numpy()
		return [{self.args.metric : [correct, total]}]

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
			return token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
				options_tensor, goldids_next_variable, goldids_prev_variable
		else:
			return token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch, \
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
		network_parameters.add_argument("--dl_classifier_input_size", type=int)
		network_parameters.add_argument("--dl_classifier_hidden_size", type=int)

