import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from src.models.factory import RegisterModel
from src.utils.utility_functions import variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.nn.util import sort_batch_by_length
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper


@RegisterModel('avg_pooling')
class AveragePool(nn.Module):
	def __init__(self, args, **kwargs):
		super(AveragePool, self).__init__()
		# self.input_size  = args.lookup_input_size
		# self.hidden_size = args.lookup_hidden_size
		# self.output_size = args.lookup_output_size
		self.kernel_size = args.lookup_kernel_size
		self.stride = args.lookup_stride

		## output  = [(input + 2*padding  - kernel_size)/stride  + 1]
		self.average_pooling = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=1)

	def forward(self, batches):
		## pooling only happens over tokens not sure of the format here
		output = self.average_pooling(input)
		return output


@RegisterModel('avg')
class Average(nn.Module):
	def __init__(self, args, **kwargs):
		super(Average, self).__init__()
		self.args = args

	def forward(self, *input):
		data, mask = input[0], input[1]
		mask = 1e-6 + mask
		output = (data * mask.unsqueeze(2)).sum(1) / mask.sum(1).unsqueeze(1)
		# output[output != output] = 0
		return output

@RegisterModel('weighted_avg')
class WightedAverage(nn.Module):
	def __init__(self, args, **kwargs):
		super(WightedAverage, self).__init__()
		self.args = args

	def forward(self, *input):
		data, mask, idfs = input[0], input[1], input[2]
		mask = 1e-6 + mask
		output = (data * mask.unsqueeze(2)* idfs.unsqueeze(2)).sum(1) / mask.sum(1).unsqueeze(1)
		# output[output != output] = 0
		return output


@RegisterModel('recurrent')
class Recurrent(nn.Module):
	def __init__(self, args, **kwargs):
		super(Recurrent, self).__init__()
		self.args = args
		self.input_size = args.embed_size
		self.hidden_size = int(args.embed_size/2)
		self.num_layers = 1
		module = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
							batch_first=True, dropout=args.dropout, bidirectional=True)
		self.lstm = PytorchSeq2VecWrapper(module)


	def forward(self, *input):
		# logic to get rid of bad sequences that have 0 lengths
		x, x_mask = input[0], input[1]
		return self.lstm(x, x_mask)
