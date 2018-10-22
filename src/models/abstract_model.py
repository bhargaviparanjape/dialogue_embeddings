from abc import ABCMeta, abstractmethod, abstractproperty
from src.utils.utility_functions import variable,FloatTensor,ByteTensor,LongTensor,select_optimizer
import torch
import os,copy,sys, logging

logger = logging.getLogger(__name__)

class AbstractModel():
	__metaclass__ = ABCMeta

	def __init__(self, args, inputs):
		self.args = args
		self.updates = 0

	def cuda(self):
		self.network = self.network.cuda()

	def update(self, inputs):
		## update based on inputs
		"""Forward a batch of examples; step the optimizer to update weights."""
		if not self.optimizer:
			raise RuntimeError('No optimizer set.')

		# Train mode
		self.network.train()

		# Run forward
		batch_size, *inputs = self.vectorize(inputs, mode = "train")
		loss = self.network(*inputs)

		# Update parameters
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.network.parameters(),
									   self.args.clip_threshold)
		self.optimizer.step()
		self.updates += 1

		# Return loss and batch size [to average over]
		if self.args.use_cuda:
			loss_value = loss.data.cpu().item()
		else:
			loss_value = loss.data.item()
		return loss_value, batch_size

	def set_vocabulary(self, vocabulary):
		raise NotImplementedError

	def parallelize(self):
		"""Use data parallel to copy the model across several gpus.
		This will take all gpus visible with CUDA_VISIBLE_DEVICES.
		"""
		self.parallel = True
		self.network = torch.nn.DataParallel(self.network)

	def init_optimizer(self):
		parameters = [p for p in self.network.parameters() if p.requires_grad]
		self.optimizer = select_optimizer(self.args, parameters)

	def vectorize(self, batch, mode = "train"):
		raise NotImplementedError

	def save(self):
		# model parameters; metrics;
		if self.args.parallel:
			network = self.network.module
		else:
			network = self.network
		state_dict = copy.copy(network.state_dict())
		# Pop layers if required
		params = {
			'word_dict': self.vocabulary,
			'args': self.args,
			'state_dict': state_dict
		}
		try:
			torch.save(params, os.path.join(self.args.model_dir, self.args.model_path))
		except BaseException:
			logger.warning('WARN: Saving failed... continuing anyway.')

	@staticmethod
	def add_args(parser):
		raise NotImplementedError

	def load(self, pretrained_model_path=None, pretrained_embed_path=None):
		if pretrained_embed_path is not None:
			trained_model = torch.load(pretrained_embed_path)
