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


@RegisterModel('elmo_lstm')
class StackedBRNNHighway(_EncoderBase):
	def __init__(self, args, **kwargs):
		super(StackedBRNNHighway, self).__init__()
		"""
		A stacked, bidirectional LSTM which uses
		:class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`'s
		with highway layers between the inputs to layers.
		The inputs to the forward and backward directions are independent - forward and backward
		states are not concatenated between layers.

		Additionally, this LSTM maintains its `own` state, which is updated every time
		``forward`` is called. It is dynamically resized for different batch sizes and is
		designed for use with non-continuous inputs (i.e inputs which aren't formatted as a stream,
		such as text used for a language modelling task, which is how stateful RNNs are typically used).
		This is non-standard, but can be thought of as having an "end of sentence" state, which is
		carried across different sentences.

		Parameters
		----------
		input_size : ``int``, required
			The dimension of the inputs to the LSTM.
		hidden_size : ``int``, required
			The dimension of the outputs of the LSTM.
		cell_size : ``int``, required.
			The dimension of the memory cell of the
			:class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`.
		num_layers : ``int``, required
			The number of bidirectional LSTMs to use.
		requires_grad: ``bool``, optional
			If True, compute gradient of ELMo parameters for fine tuning.
		recurrent_dropout_probability: ``float``, optional (default = 0.0)
			The dropout probability to be used in a dropout scheme as stated in
			`A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
			<https://arxiv.org/abs/1512.05287>`_ .
		state_projection_clip_value: ``float``, optional, (default = None)
			The magnitude with which to clip the hidden_state after projecting it.
		memory_cell_clip_value: ``float``, optional, (default = None)
			The magnitude with which to clip the memory cell.
		"""
		self.args = args
		self.input_size = args.embed_size
		self.hidden_size = args.hidden_size
		self.num_layers = args.num_layers
		self.cell_size = args.cell_size
		self.dropout = args.dropout
		# self.requires_grad = args.requires_grad

		forward_layers = []
		backward_layers = []

		# Original ELMo options file contents
		'''
		{"lstm": {"use_skip_connections": true, "projection_dim": 512, "cell_clip": 3, "proj_clip": 3, "dim": 4096,
				  "n_layers": 2}, "char_cnn": {"activation": "relu",
											   "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512],
														   [7, 1024]], "n_highway": 2, "embedding": {"dim": 16},
											   "n_characters": 262, "max_characters_per_token": 50}}
		'''

		lstm_input_size = self.input_size
		go_forward = True
		for layer_index in range(self.num_layers):
			forward_layer = LstmCellWithProjection(lstm_input_size,
													self.hidden_size,
													self.cell_size,
													go_forward,
                                                   self.dropout,
													3,
													3)
			backward_layer = LstmCellWithProjection(lstm_input_size,
													self.hidden_size,
													self.cell_size,
													not go_forward,
													self.dropout,
													3,
													3)
			lstm_input_size = self.hidden_size

			self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
			self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
			forward_layers.append(forward_layer)
			backward_layers.append(backward_layer)
		self.forward_layers = forward_layers
		self.backward_layers = backward_layers



	def forward(self, input, mask):
		## if _states
		# if self._states is not None and self._states[0].size(1)
		if not self.training:
			self.reset_states()
		batch_size, total_sequence_length = mask.size()
		stacked_sequence_output, final_states, restoration_indices = \
			self.sort_and_run_forward(self._lstm_forward, input, mask)

		num_valid, returned_timesteps, num_directions, encoder_dim = stacked_sequence_output.size()
		# Add back invalid rows which were removed in the call to sort_and_run_forward.
		if num_valid < batch_size:
			zeros = stacked_sequence_output.new_zeros(batch_size - num_valid,
			                                          returned_timesteps,
			                                          num_directions,
			                                          encoder_dim)
			stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 0)

			# The states also need to have invalid rows added back.
			new_states = []
			num_layers = final_states[0].size(0)
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
			zeros = stacked_sequence_output.new_zeros(batch_size,
			                                          sequence_length_difference,
			                                          num_directions,
			                                          stacked_sequence_output[0].size(-1))
			stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

		# Is this step just book keeping or contributing to gradient
		# self._update_states(final_states, restoration_indices)

		# Restore the original indices and return the sequence.
		# Has shape (batch_size, sequence_length, num_directions, hidden_size)
		# Convert to (batch_size, sequence_length, num_directions*hidden_size) for compatibility for next code
		restored_stacked_output = stacked_sequence_output.index_select(0, restoration_indices)
		# reshaped_stacked_output = restored_stacked_output.view(restored_stacked_output.size(0), restored_stacked_output.size(1), -1)
		return restored_stacked_output, final_states


	def _lstm_forward(self, inputs, initial_state = None):
		"""
		Parameters
		----------
		inputs : ``PackedSequence``, required.
			A batch first ``PackedSequence`` to run the stacked LSTM over.
		initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
			A tuple (state, memory) representing the initial hidden state and memory
			of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
			(num_layers, batch_size, 2 * cell_size) respectively.

		Returns
		-------
		output_sequence : ``torch.FloatTensor``
			The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
		final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
			The per-layer final (state, memory) states of the LSTM, with shape
			(num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
			respectively. The last dimension is duplicated because it contains the state/memory
			for both the forward and backward layers.
		"""
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

			# sequence_outputs.append(torch.cat([forward_output_sequence,backward_output_sequence], -1))
			sequence_outputs.append(torch.cat([forward_output_sequence.unsqueeze(2), backward_output_sequence.unsqueeze(2)], 2))
			# Append the state tuples in a list, so that we can return
			# the final states for all the layers.
			final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
								 torch.cat([forward_state[1], backward_state[1]], -1)))

		# stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
		stacked_sequence_outputs = torch.cat(sequence_outputs, 3)
		# Stack the hidden state and memory for each layer into 2 tensors of shape
		# (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
		# respectively.
		final_hidden_states, final_memory_states = zip(*final_states)
		final_state_tuple = (torch.cat(final_hidden_states, 0), torch.cat(final_memory_states, 0))
		# Forward and backward need to be separated and the differnet layers have to be concatenated
		return stacked_sequence_outputs, final_state_tuple
