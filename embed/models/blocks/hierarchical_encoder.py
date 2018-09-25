import torch
from torch.nn.modules import Dropout
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from allennlp.common.checks import ConfigurationError
from embed.models.factory import RegisterModel, variable, FloatTensor, ByteTensor, LongTensor
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection
from allennlp.modules.encoder_base import _EncoderBase

@RegisterModel('hierarchical_encoder')
class HierarchicalEncoder(_EncoderBase):
	def __init__(self, args):
		super(HierarchicalEncoder, self).__init__()
		self.args = args
		self.dropout = args.elmo_dropout
		self.input_size = args.elmo_input_size
		self.hidden_size = args.elmo_hidden_size
		self.num_layers = args.elmo_num_layers
		self.cell_size = args.elmo_cell_size
		self.requires_grad = args.elmo_requires_grad

		forward_layers = []
		backward_layers = []

		lstm_input_size = self.input_size
		go_forward = True
		for layer_index in range(self.num_layers):
			forward_layer = LstmCellWithProjection(lstm_input_size,
												   self.hidden_size,
												   self.cell_size,
												   go_forward,
												   self.dropout,
												   None, None)
			backward_layer = LstmCellWithProjection(lstm_input_size,
													self.hidden_size,
													self.cell_size,
													not go_forward,
													self.dropout,
													None, None)
			lstm_input_size = self.hidden_size

			self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
			self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
			forward_layers.append(forward_layer)
			backward_layers.append(backward_layer)
		self.forward_layers = forward_layers
		self.backward_layers = backward_layers




	def forward(self, inputs, mask):
		batch_size, total_sequence_length = mask.size()
		stacked_sequence_output, final_states, restoration_indices = \
			self.sort_and_run_forward(self._lstm_forward, inputs, mask)

		num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
		# Add back invalid rows which were removed in the call to sort_and_run_forward.
		if num_valid < batch_size:
			zeros = stacked_sequence_output.new_zeros(num_layers,
													  batch_size - num_valid,
													  returned_timesteps,
													  encoder_dim)
			stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

			# The states also need to have invalid rows added back.
			new_states = []
			for state in final_states:
				state_dim = state.size(-1)
				zeros = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
				new_states.append(torch.cat([state, zeros], 1))
			final_states = new_states

		# It's possible to need to pass sequences which are padded to longer than the
		# max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
		# the sequences mean that the returned tensor won't include these dimensions, because
		# the RNN did not need to process them. We add them back on in the form of zeros here.
		sequence_length_difference = total_sequence_length - returned_timesteps
		if sequence_length_difference > 0:
			zeros = stacked_sequence_output.new_zeros(num_layers,
													  batch_size,
													  sequence_length_difference,
													  stacked_sequence_output[0].size(-1))
			stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

		self._update_states(final_states, restoration_indices)

		# Restore the original indices and return the sequence.
		# Has shape (num_layers, batch_size, sequence_length, hidden_size)
		return stacked_sequence_output.index_select(1, restoration_indices)

	def _lstm_forward(self,
					  inputs: PackedSequence,
					  initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
			Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
		if initial_state is None:
			hidden_states: List[Optional[Tuple[torch.Tensor,
											   torch.Tensor]]] = [None] * len(self.forward_layers)
		elif initial_state[0].size()[0] != len(self.forward_layers):
			raise ConfigurationError("Initial states were passed to forward() but the number of "
									 "initial states does not match the number of layers.")
		else:
			hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

		inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
		forward_output_sequence = inputs
		backward_output_sequence = inputs

		final_states = []
		sequence_outputs = []
		for layer_index, state in enumerate(hidden_states):
			forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
			backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

			forward_cache = forward_output_sequence
			backward_cache = backward_output_sequence

			if state is not None:
				forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
				forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
				forward_state = (forward_hidden_state, forward_memory_state)
				backward_state = (backward_hidden_state, backward_memory_state)
			else:
				forward_state = None
				backward_state = None

			forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
																   batch_lengths,
																   forward_state)
			backward_output_sequence, backward_state = backward_layer(backward_output_sequence,
																	  batch_lengths,
																	  backward_state)
			# Skip connections, just adding the input to the output.
			if layer_index != 0:
				forward_output_sequence += forward_cache
				backward_output_sequence += backward_cache

			sequence_outputs.append(torch.cat([forward_output_sequence,
											   backward_output_sequence], -1))
			# Append the state tuples in a list, so that we can return
			# the final states for all the layers.
			final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
								 torch.cat([forward_state[1], backward_state[1]], -1)))

		stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
		# Stack the hidden state and memory for each layer into 2 tensors of shape
		# (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
		# respectively.
		final_hidden_states, final_memory_states = zip(*final_states)
		final_state_tuple: Tuple[torch.FloatTensor,
								 torch.FloatTensor] = (torch.cat(final_hidden_states, 0),
													   torch.cat(final_memory_states, 0))
		return stacked_sequence_outputs, final_state_tuple