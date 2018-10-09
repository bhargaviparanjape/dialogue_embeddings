import torch
import torch.nn as nn
import numpy as np
import logging
from torch.nn import functional as F
from src.models import factory as model_factory
from src.models.factory import RegisterModel
import json

logger = logging.getLogger(__name__)

#############################################################
############# GENERAL DIALOGUE EMBEDDINGS ###################
#############################################################
@RegisterModel('dl_embedder')
class DialogueEmbedder(nn.Module):
	def __init__(self, args, **kwargs):
		super(DialogueEmbedder, self).__init__()
		self.args = args
		if not args.fixed_token_encoder:
			self.token_encoder = model_factory.get_model_by_name(args.token_encoder, args)
		if not args.fixed_utterance_encoder:
			self.utterance_encoder = model_factory.get_model_by_name(args.utterance_encoder, args)
		conversation_dict = {"input_size" : args.embed_size, "hidden_size" : args.hidden_size, "num_layers": args.num_layers}
		self.conversation_encoder = model_factory.get_model_by_name(args.conversation_encoder, args, kwargs = conversation_dict)


	def forward(self, *input):
		[token_embeddings, input_mask_variable, conversation_mask, max_num_utterances_batch] = input[0]
		if self.args.fixed_utterance_encoder:
			utterance_encodings = token_embeddings
		else:
			utterance_encodings = self.utterance_encoder(token_embeddings, input_mask_variable)

		# Reshape batch to run sequence over conversations
		conversation_batch_size = int(utterance_encodings.shape[0] / max_num_utterances_batch)
		reshaped_utterances = utterance_encodings.view(conversation_batch_size, max_num_utterances_batch, utterance_encodings.shape[1])

		# Encode Conversation
		conversation_encoded, _ = self.conversation_encoder(reshaped_utterances, conversation_mask)

		# Reshape conversation back to batch
		conversation_encoded_flattened = conversation_encoded.view(conversation_batch_size * max_num_utterances_batch,
																   -1, conversation_encoded.shape[2])
		return conversation_encoded_flattened


	@staticmethod
	def add_args(parser):
		model_parameters = parser.add_argument_group("Model Parameters")
		model_parameters.add_argument("--hidden-size", type=int)
		model_parameters.add_argument("--num-layers", type=int)