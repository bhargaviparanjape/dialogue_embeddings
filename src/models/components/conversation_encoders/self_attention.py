import torch
from torch.nn.modules import Dropout
import torch.nn as nn
import torch.nn.functional as F

from src.models.factory import RegisterModel


@RegisterModel('elmo_lstm')
class SelfAttention(nn.Module):
	def __init__(self, args):
		super(SelfAttention, self).__init__()


	def forward(self, input, mask):

		pass