import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.factory import RegisterModel
from src.utils.utility_functions import variable, FloatTensor, ByteTensor, LongTensor

@RegisterModel('lstm_decoder')
class LSTMDecoder(nn.Module):
	def __init__(self, args, **kwargs):
		super(LSTMDecoder, self).__init__()
		self.args = args
		self.input_size = args.embed_size
		self.hidden_size = args.hidden_size
		self.num_layers = args.num_layers
		self.output_size = kwargs["kwargs"]["output_size"]
		self.network = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
							batch_first=True, dropout=args.dropout, bidirectional=False)
		self.out = nn.Linear(self.hidden_size, self.output_size)


	def forward(self, input, hidden, mask, teacher_forcing = True):
		## given hidden representation and techer forcing at test time (greedy or beam decoding)
		target_length = input.shape[1]
		decoder_input = input[:,0,:]
		decoder_hidden = hidden
		if teacher_forcing:
			for y in range(1,target_length):
				decoder_output, decoder_hidden = self.lstm(decoder_input, decoder_hidden)
				## process decoder_output to get the embeddings of the maximum vocabulary elements in the output (or get likelihood objectives)
				decoder_input = input[:, y, :]
