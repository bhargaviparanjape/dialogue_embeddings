import torch
from torch.nn.modules import Dropout
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder

from src.models.factory import RegisterModel

@RegisterModel('transformer')
class Transformer(nn.Module):
	def __init__(self, args):
		super(Transformer, self).__init__()
		## Define StackedSelfAttentionAllenNLP
		self.args = args
		## input_dim: int,
		self.input_size = args.embed_size
		# hidden_dim: int,
		self.hidden_size = args.hidden_size
		# projection_dim: int, (256)
		projection_dim = 256
		# feedforward_hidden_dim: int, (512)
		feedforward_hidden_dim = 512
		# num_layers: int, (2)
		self.num_layers = args.num_layers
		# num_attention_heads: int, (8)
		self.num_heads = args.num_heads
		# use_positional_encoding: bool = True,
		# dropout_prob: float = 0.1,
		# residual_dropout_prob: float = 0.2,
		# attention_dropout_prob: float = 0.1)
		self.encoder = StackedSelfAttentionEncoder(
			input_dim = self.input_size,
		    hidden_dim = self.hidden_size,
			projection_dim=projection_dim,
			feedforward_hidden_dim = feedforward_hidden_dim,
			num_layers =  self.num_layers,
			num_attention_heads = self.num_heads
		)

	def forward(self, input, mask):
		encoded_output =  self.encoder(input, mask)

