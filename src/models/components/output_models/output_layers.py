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
		# self.input_size = args.hidden_size * 2
		# self.hidden_size = args.output_hidden_size
		# self.output_size = args.output_size
		self.network = nn.Sequential(
			nn.Linear(self.input_size, self.hidden_size),
			nn.ReLU(),
			nn.Linear(self.hidden_size, self.output_size),
		)

	def forward(self, input):
		return self.network(input)