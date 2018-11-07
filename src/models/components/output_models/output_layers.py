import torch
import torch.nn as nn
from src.models.factory import RegisterModel
from torch.nn import functional as F
import logging
from src.models import factory as model_factory

logger = logging.getLogger(__name__)

@RegisterModel('mlp')
class MultiLayerPerceptron(nn.Module):
	def __init__(self, args, **kwargs):
		super(MultiLayerPerceptron, self).__init__()
		self.input_size = kwargs["kwargs"]["input_size"]
		self.hidden_size = kwargs["kwargs"]["hidden_size"]
		self.output_size = kwargs["kwargs"]["output_size"]
		self.num_layers = kwargs["kwargs"]["num_layers"]

		self.network = nn.ModuleList()
		for i in range(self.num_layers):
			input_size = self.input_size if i == 0 else self.hidden_size
			output_size = self.output_size if i == self.num_layers - 1 else self.hidden_size
			if i != self.num_layers - 1:
				self.network.append(nn.Sequential(
					nn.Linear(input_size, output_size),
					nn.ReLU())
				)
			else:
				self.network.append(nn.Linear(input_size, output_size))

	def forward(self, input):
		for i in range(self.num_layers):
			output = self.network[i](input)
			input = output
		return output
