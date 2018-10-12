from abc import ABCMeta, abstractmethod, abstractproperty
import torch

class AbstractModel():
	__metaclass__ = ABCMeta

	def __init__(self, args, inputs):
		self.args = args

	def cuda(self):
		raise NotImplementedError

	def set_vocabulary(self, vocabulary):
		raise NotImplementedError


	def vectorize(self, *inputs):
		raise NotImplementedError

	@staticmethod
	def add_args(parser):
		raise NotImplementedError

	def load(self, pretrained_model_path=None, pretrained_embed_path=None):
		if pretrained_embed_path is not None:
			trained_model = torch.load(pretrained_embed_path)
