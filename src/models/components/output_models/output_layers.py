import torch
import torch.nn as nn
from src.models.factory import RegisterModel
from torch.nn import functional as F
import logging
from src.models import factory as model_factory
import allennlp.nn.util as util

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


@RegisterModel('linear_crf')
class LinearCRF(nn.Module):
	def __init__(self, args, **kwargs):
		super(LinearCRF, self).__init__()
		self.args = args
		self.num_tags = kwargs["kwargs"]["output_size"]

		self.transitions = torch.nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))

		## Add constraints to transitions
		constraint_mask = torch.Tensor(self.num_tags + 2, self.num_tags + 2).fill_(1.)
		self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_normal_(self.transitions)

	def _input_likelihood(self, logits, mask):
		"""
		Computes the (batch_size,) denominator term for the log-likelihood, which is the
		sum of the likelihoods across all possible state sequences.
		"""
		batch_size, sequence_length, num_tags = logits.size()

		# Transpose batch size and sequence dimensions
		mask = mask.float().transpose(0, 1).contiguous()
		logits = logits.transpose(0, 1).contiguous()
		alpha = logits[0]
		for i in range(1, sequence_length):
			emit_scores = logits[i].view(batch_size, 1, num_tags)
			transition_scores = self.transitions.view(1, num_tags, num_tags)
			broadcast_alpha = alpha.view(batch_size, num_tags, 1)
			inner = broadcast_alpha + emit_scores + transition_scores
			alpha = (util.logsumexp(inner, 1) * mask[i].view(batch_size, 1) +
			         alpha * (1 - mask[i]).view(batch_size, 1))

		stops = alpha
		return util.logsumexp(stops)

	def _joint_likelihood(self, logits, labels, mask):
		batch_size, sequence_length, _ = logits.data.shape
		logits = logits.transpose(0, 1).contiguous()
		mask = mask.float().transpose(0, 1).contiguous()
		tags = labels.transpose(0, 1).contiguous().squeeze(-1)
		score = 0.0
		for i in range(sequence_length - 1):
			current_tag, next_tag = tags[i], tags[i + 1]
			transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
			emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
			score = score + transition_score * mask[i + 1] + emit_score * mask[i]
		last_tag_index = mask.sum(0).long() - 1
		last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
		last_transition_score = 0.0
		last_inputs = logits[-1]  # (batch_size, num_tags)
		last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
		last_input_score = last_input_score.squeeze()  # (batch_size,)

		score = score + last_transition_score + last_input_score * mask[-1]

		return score


	def forward(self, *input):
		[logits, labels, mask] = input
		batch_size = logits.size(0)
		log_denominator = self._input_likelihood(logits, mask)
		log_numerator = self._joint_likelihood(logits, labels, mask)
		return torch.sum(log_denominator - log_numerator)/batch_size


	def _viterbi_decode(self, logits, mask):
		_, max_seq_length, num_tags = logits.size()
		logits, mask = logits.data, mask.data
		start_tag = num_tags
		end_tag = num_tags + 1
		transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

		constrained_transitions = (
			self.transitions * self._constraint_mask[:num_tags, :num_tags] +
			-10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
		)

		transitions[:num_tags, :num_tags] = constrained_transitions.data
		transitions[start_tag, :num_tags] = (-10000.0 *
		                                     (1 - self._constraint_mask[start_tag, :num_tags].detach()))
		transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
		best_paths = []
		# Pad the max sequence length by 2 to account for start_tag + end_tag.
		tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

		for prediction, prediction_mask in zip(logits, mask):
			sequence_length = torch.sum(prediction_mask).long()

			# Start with everything totally unlikely
			tag_sequence.fill_(-10000.)
			# At timestep 0 we must have the START_TAG
			tag_sequence[0, start_tag] = 0.
			# At steps 1, ..., sequence_length we just use the incoming prediction
			tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
			# And at the last timestep we must have the END_TAG
			tag_sequence[sequence_length + 1, end_tag] = 0.
			# We pass the tags and the transitions to ``viterbi_decode``.
			viterbi_path, viterbi_score = util.viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
			# Get rid of START and END sentinels and append.
			viterbi_path = viterbi_path[1:-1]
			best_paths.append((viterbi_path + [0]*(max_seq_length - len(viterbi_path)), viterbi_score.item()))


		return best_paths