import torch
from torch.nn.modules import Dropout
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.nn import Dropout, Linear

from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from src.models.factory import RegisterModel

@RegisterModel("multi_head_self_attention")
class MultiHeadSelfAttention(Seq2SeqEncoder):
	def __init__(self):
		super(MultiHeadSelfAttention, self).__init__()
