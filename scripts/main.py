import sys,logging,argparse
from os.path import dirname, realpath
import numpy as np
import torch
import random
sys.path.append(dirname(dirname(realpath(__file__))))
from embed.utils import parameters
from embed.dataloaders import factory as dataloader_factory
from embed.models import factory as model_factory
from embed.learn.train import evaluate, train, generate_embeddings

if __name__ == '__main__':

	args = parameters.parse_arguments()

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger(__name__)

	print(args)

	dataset = dataloader_factory.get_dataset(args,logger)
	vars(args)["vocabulary"] = dataset.vocabulary.vocabulary
	model = model_factory.get_model(args, args.model, logger)

	if args.run_mode == "train":
		train(args, dataset, model,logger)
		dataset = dataloader_factory.get_dataset(args, logger)
		# generate_embeddings(args, dataset, model, logger)
	elif args.run_mode == "test":
		model = torch.load(args.model_path)
		evaluate(args, dataset, model,logger)

