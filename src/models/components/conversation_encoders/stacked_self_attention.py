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
		## Define StackedSelfAttentionAllenNLP
		pass

	def forward(self, input):
		pass
