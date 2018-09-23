import torch
from torch.autograd import Variable

NO_MODEL_ERR = "Model {} not in MODEL_REGISTRY! Available models are {}"

MODEL_REGISTRY = {}

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


def variable(v, arg_use_cuda=True, volatile=False):
    if torch.cuda.is_available() and arg_use_cuda:
        return Variable(v, volatile=volatile).cuda()
    return Variable(v, volatile=volatile)


def RegisterModel(model_name):
	"""Registers a model."""

	def decorator(f):
		MODEL_REGISTRY[model_name] = f
		return f

	return decorator


def get_embeddings(args, embedding_layer_name, logger=None):
	if embedding_layer_name not in MODEL_REGISTRY:
		raise Exception(
			NO_MODEL_ERR.format(embedding_layer_name, MODEL_REGISTRY.keys()))
	if embedding_layer_name in MODEL_REGISTRY:
		model = MODEL_REGISTRY[embedding_layer_name](args)
	return model


def get_model(args, model_name, logger=None):
	if model_name not in MODEL_REGISTRY:
		raise Exception(
			NO_MODEL_ERR.format(model_name, MODEL_REGISTRY.keys()))

	if model_name in MODEL_REGISTRY:
		model = MODEL_REGISTRY[model_name](args)
	if args.use_cuda:
		model = model.cuda()
	return model


NO_OBJECTIVE_ERR = "Objective {} not in OBJECTIVE_REGISTRY! Available objectives are {}"

OBJECTIVE_REGISTRY = {}

def RegisterObjective(objective_name):
	"""Registers a model."""

	def decorator(f):
		OBJECTIVE_REGISTRY[objective_name] = f
		return f

	return decorator


def get_objective(args, objective_name, logger):
	if objective_name not in OBJECTIVE_REGISTRY:
		raise Exception(
			NO_OBJECTIVE_ERR.format(objective_name, OBJECTIVE_REGISTRY.keys()))

	if objective_name in OBJECTIVE_REGISTRY:
		objective = OBJECTIVE_REGISTRY[objective_name](args)

	return objective
