import sys, logging, argparse, json
from os.path import dirname, realpath
import numpy as np
import torch
import random
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from src.utils import global_parameters
from src.learn import config as train_config
from src.models import config as model_config

from src.dataloaders import factory as data_factroy
from src.models import factory as model_factory

from src.learn import train

logger = logging.getLogger()

def init_model(args, dataset):
	data_factroy.set_dataset_arguments(args, dataset)
	model = model_factory.get_model(args)
	if args.pretrained_dialogue_embed_path is not None:
		model.load(pretrained_embed=args.pretrained_dialogue_embed_path)
	model.set_vocabulary(dataset.vocabulary.vocabulary)
	return model


def main(args):

	## DATA
	logger.info('-' * 100)
	logger.info('Load data files')
	dataset = data_factroy.get_dataset(args, logger)
	# TODO: Replace current dataloder with pytorch dataloader, batching function

	## MODEL
	logger.info('-' * 100)
	# TODO: Checkpoint, Pre-trained Load
	logger.info('Building model from scratch...')
	model = init_model(args, dataset)

	# Setup optimizer
	model.init_optimizer()

	# Use GPU
	if args.use_cuda:
		model.cuda()

	# Use multiple GPUs
	if args.use_cuda and args.parallel:
		model.parallelize()

	# Train Model
	train.train_epochs(args, dataset, model)

	# Predict on Test Set



if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		'Conversation Embeddings',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	# General Arguments + File Paths + Dataset Paths
	global_parameters.add_args(parser)

	# Train Arguments
	train_config.add_args(parser)

	# Model Arguments
	model_config.add_args(parser)

	args = parser.parse_args()

	# Set CUDA
	if args.cuda and torch.cuda.is_available():
		vars(args)['use_cuda'] = True
	else:
		vars(args)['use_cuda'] = False

	# Set Random Seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	# Set Logging
	logger.setLevel(logging.DEBUG)
	fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
	console = logging.StreamHandler()
	console.setFormatter(fmt)
	logger.addHandler(console)
	if args.log_file:
		logfile = logging.FileHandler(args.log_file, 'w')
		logfile.setFormatter(fmt)
		logger.addHandler(logfile)

	logger.info('COMMAND: %s' % ' '.join(sys.argv))

	# PRINT CONFIG
	logger.info('-' * 100)
	logger.info('CONFIG:\n%s' %
				json.dumps(vars(args), indent=4, sort_keys=True))

	main(args)

