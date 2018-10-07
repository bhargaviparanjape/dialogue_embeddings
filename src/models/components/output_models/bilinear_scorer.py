import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from src.models.factory import RegisterModel


@RegisterModel('bilinear')
class BiLinear(nn.Module):
	def __init__(self, args, logger):
		self.args = args
		self.network = nn.Bilinear(args.bilinear_input_size, args.bilinear_hidden_size, args.bilinear_output_size)

	def forward(self, *input):
		[input1, input2] = input[0]
		logits = self.next_utterance_scorer(input1, input2)
		return logits

	def add_argument(self, parser):
		output_parameters = parser.add_argument_group("Output Layer Parameters")
		output_parameters.add_argument("bilinear_input_size")
		output_parameters.add_argument("bilinear_hidden_size")
		output_parameters.add_argument("bilinear_output_size", default = 1)
