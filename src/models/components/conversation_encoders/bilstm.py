import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.factory import RegisterModel
from src.utils.utility_functions import variable, FloatTensor, ByteTensor, LongTensor


@RegisterModel('bilstm')
class BiLSTMEncoder(nn.Module):
	def __init__(self, args, **kwargs):
		super(BiLSTMEncoder, self).__init__()
		self.input_size = kwargs["input_size"]
		self.hidden_size = kwargs["hidden_size"]
		self.num_layers = kwargs["num_layers"]
		# self.input_size = args.embed_size
		# self.hidden_size = args.hidden_size
		# self.num_layers = args.num_layers
		self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
							batch_first=True, dropout=args.dropout, bidirectional=True)

	def forward(self, x, x_mask):
		lengths = x_mask.data.eq(1).long().sum(1)
		_, idx_sort = torch.sort(lengths, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)
		lengths = list(lengths[idx_sort])
		idx_sort = variable(idx_sort)
		idx_unsort = variable(idx_unsort)

		# Sort x
		x = x.index_select(0, idx_sort)

		rnn_input = pack_padded_sequence(x, lengths, batch_first=True)
		self.lstm.flatten_parameters()
		outputs, (hidden, cell) = self.lstm(rnn_input)
		outputs_unpacked, _ = pad_packed_sequence(outputs, batch_first=True)
		outputs_unpacked = outputs_unpacked[idx_unsort]
		## hidden and cell are still sorted ordered
		return outputs_unpacked, hidden

	# @staticmethod
	# def add_args(parser):
	# 	model_parameters = parser.add_argument_group("BiLSTM Parameters")
	# 	model_parameters.add_argument("--hidden-size", type=int)
	# 	model_parameters.add_argument("--num-layers", type=int)

@RegisterModel('lstmdecoder')
class LSTMDecoder(nn.Module):
	def __init__(self, args):
		raise NotImplementedError

	def forward(self, *input):
		## given hidden representation and techer forcing at test time (greedy or beam decoding)
		raise NotImplementedError

