import torch
from torch.nn.modules import Dropout
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from allennlp.common.checks import ConfigurationError
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection
from allennlp.modules.encoder_base import _EncoderBase

from src.models.factory import RegisterModel
from src.utils.utility_functions import FloatTensor, ByteTensor, LongTensor, variable

@RegisterModel('stacked_bilstm')
class StackedBRNN(nn.Module):
	"""Stacked Bi-directional RNNs.

	Differs from standard PyTorch library in that it has the option to save
	and concat the hidden states between layers. (i.e. the output hidden size
	for each sequence input is num_layers * hidden_size).
	"""
	def __init__(self, args):
		super(StackedBRNN, self).__init__()
		self.padding = True
		self.dropout_output = False
		self.dropout_rate = args.dropout
		self.input_size = args.encoder_input_size
		self.hidden_size = args.encoder_hidden_size
		self.num_layers = args.encoder_num_layers
		self.concat_layers = True
		self.rnns = nn.ModuleList()
		for i in range(self.num_layers):
			self.input_size = self.input_size if i == 0 else 2 * self.hidden_size
			self.rnns.append(nn.LSTM(self.input_size, self.hidden_size,
									  num_layers=1,
									  bidirectional=True))

	def forward(self, x, x_mask):
		"""Encode either padded or non-padded sequences.

		Can choose to either handle or ignore variable length sequences.
		Always handle padding in eval.

		Args:
			x: batch * len * hdim
			x_mask: batch * len (1 for padding, 0 for true)
		Output:
			x_encoded: batch * len * hdim_encoded
		"""
		if x_mask.data.sum() == 0:
			# No padding necessary.
			output,_ = self._forward_unpadded(x, x_mask)
		elif self.padding or not self.training:
			# Pad if we care or if its during eval.
			output,_ = self._forward_padded(x, x_mask)
		else:
			# We don't care.
			output,_ = self._forward_unpadded(x, x_mask)

		return output.contiguous(), None

	def _forward_unpadded(self, x, x_mask):
		"""Faster encoding that ignores any padding."""
		# Transpose batch and sequence dims
		x = x.transpose(0, 1)

		# Encode all layers
		outputs = [x]
		for i in range(self.num_layers):
			rnn_input = outputs[-1]

			# Apply dropout to hidden input
			if self.dropout_rate > 0:
				rnn_input = F.dropout(rnn_input,
									  p=self.dropout_rate,
									  training=self.training)
			# Forward
			rnn_output = self.rnns[i](rnn_input)[0]
			outputs.append(rnn_output)

		# Concat hidden layers
		if self.concat_layers:
			output = torch.cat(outputs[1:], 2)
		else:
			output = outputs[-1]

		# Transpose back
		output = output.transpose(0, 1)

		# Dropout on output layer
		if self.dropout_output and self.dropout_rate > 0:
			output = F.dropout(output,
							   p=self.dropout_rate,
							   training=self.training)
		return output, None

	def _forward_padded(self, x, x_mask):
		"""Slower (significantly), but more precise, encoding that handles
		padding.
		"""
		# Compute sorted sequence lengths
		lengths = x_mask.data.eq(1).long().sum(1)
		_, idx_sort = torch.sort(lengths, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)

		lengths = list(lengths[idx_sort])
		idx_sort = variable(idx_sort)
		idx_unsort = variable(idx_unsort)

		# Sort x
		x = x.index_select(0, idx_sort)

		# Transpose batch and sequence dims
		x = x.transpose(0, 1)

		# Pack it up
		rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

		# Encode all layers
		outputs = [rnn_input]
		for i in range(self.num_layers):
			rnn_input = outputs[-1]

			# Apply dropout to input
			if self.dropout_rate > 0:
				dropout_input = F.dropout(rnn_input.data,
										  p=self.dropout_rate,
										  training=self.training)
				rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
														rnn_input.batch_sizes)
			outputs.append(self.rnns[i](rnn_input)[0])

		# Unpack everything
		for i, o in enumerate(outputs[1:], 1):
			outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

		# Concat hidden layers or take final
		if self.concat_layers:
			output = torch.cat(outputs[1:], 2)
		else:
			output = outputs[-1]

		# Transpose and unsort
		output = output.transpose(0, 1)
		output = output.index_select(0, idx_unsort)

		# Pad up to original batch sequence length
		if output.size(1) != x_mask.size(1):
			padding = torch.zeros(output.size(0),
								  x_mask.size(1) - output.size(1),
								  output.size(2)).type(output.data.type())
			output = torch.cat([output, variable(padding)], 1)

		# Dropout on output layer
		if self.dropout_output and self.dropout_rate > 0:
			output = F.dropout(output,
							   p=self.dropout_rate,
							   training=self.training)
		# hidden representation is not exposed
		return output, None