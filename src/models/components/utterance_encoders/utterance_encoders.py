import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from src.models.factory import RegisterModel


@RegisterModel('avg_pooling')
class AveragePool(nn.Module):
	def __init__(self, args):
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
	def __init__(self, args):
		super(Average, self).__init__()
		# self.input_size  = args.lookup_input_size
		# self.hidden_size = args.lookup_hidden_size
		# self.output_size = args.lookup_output_size
		self.args = args

	def forward(self, *input):
		## pooling only happens over tokens not sure of the format here
		## mean should take mask into account
		data, mask = input[0], input[1]
		output = (data * mask.unsqueeze(2)).sum(1) / mask.sum(1).unsqueeze(1)
		output[output != output] = 0
		return output