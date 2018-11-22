import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from src.models.factory import RegisterModel


@RegisterModel('bilinear')
class BiLinear(nn.Module):
	def __init__(self, args, **kwargs):
		super(BiLinear, self).__init__()
		self.args = args
		# self.network = nn.Bilinear(args.bilinear_input_size, args.bilinear_hidden_size, args.bilinear_output_size)
		bilinear_input_size = kwargs["kwargs"]["input_size"]
		bilinear_output_size = kwargs["kwargs"]["output_size"]
		self.network = nn.Bilinear(bilinear_input_size, bilinear_input_size, bilinear_output_size)

	def forward(self, *input):
		[input1, input2] = input
		logits = self.network(input1, input2)
		return logits

	@staticmethod
	def add_args(parser):
		pass
		# output_parameters = parser.add_argument_group("Output Layer Parameters")
		# output_parameters.add_argument("--bilinear-input-size")
		# output_parameters.add_argument("--bilinear-hidden-size")
		# output_parameters.add_argument("--bilinear-output-size", default = 1)
