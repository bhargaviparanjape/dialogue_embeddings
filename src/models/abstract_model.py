from abc import ABCMeta, abstractmethod, abstractproperty

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