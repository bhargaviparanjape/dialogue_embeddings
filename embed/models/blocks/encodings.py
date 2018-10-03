import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from embed.models.factory import RegisterModel
from allennlp.commands.elmo import ElmoEmbedder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor


@RegisterModel('bilstm')
class BiLSTMEncoder(nn.Module):
	def __init__(self, args):
		super(BiLSTMEncoder, self).__init__()
		self.input_size = args.encoder_input_size
		self.hidden_size = args.encoder_hidden_size
		self.num_layers = args.encoder_num_layers
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

		rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
		self.lstm.flatten_parameters()
		outputs, (hidden, cell) = self.lstm(rnn_input)
		outputs_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
		outputs_unpacked = outputs_unpacked[idx_unsort]
		## hidden and cell are still sorted ordered
		return outputs_unpacked, hidden



# @RegisterModel('bilstm')
# class BiLSTMEncoder(nn.Module):
# 	def __init__(self, args):
# 		super(BiLSTMEncoder, self).__init__()
# 		self.input_size = args.encoder_input_size
# 		self.hidden_size = args.encoder_hidden_size
# 		self.num_layers = args.encoder_num_layers
# 		self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
# 							batch_first=True, dropout=args.dropout, bidirectional=True)
#
# 	def forward(self, batch, batch_lengths):
# 		## input shape (batch, sequence_length(no. of utterances), embeddings)
# 		packed = torch.nn.utils.rnn.pack_padded_sequence(batch, batch_lengths, batch_first=True)
# 		self.lstm.flatten_parameters()
# 		outputs, (hidden, cell) = self.lstm(packed)
# 		outputs_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
# 		return outputs_unpacked, hidden

@RegisterModel('lstmdecoder')
class LSTMDecoder(nn.Module):
	def __init__(self, args):
		raise NotImplementedError

	def forward(self, *input):
		## given hidden representation and techer forcing at test time (greedy or beam decoding)
		raise NotImplementedError

