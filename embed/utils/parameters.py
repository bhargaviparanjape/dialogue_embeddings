from collections import defaultdict
import argparse
import torch
import logging
from embed.dataloaders.SwitchBoard import DAMSL_TAGSET

def parse_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument("--cuda", action="store_true", default=True)
	parser.add_argument("--run_mode", type=str, default="train")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--truncate_dataset", action="store_true", default=False)
	parser.add_argument("--model_path", type=str, default=None)
	parser.add_argument("--embedding_path", type=str, default=None)
	parser.add_argument("--output_path", type=str, default=None)
	parser.add_argument("--logging_path", type=str, default=None)
	parser.add_argument("--log_level", type=str, default=logging.INFO)
	parser.add_argument("--datasets", type=str, default="swda")
	parser.add_argument("--dataset_paths", type=str, default="../datasets/swda/swda")
	parser.add_argument("--batch_function", type=str, default="conversation_length")

	parser.add_argument("--model", type=str, default="dialogue_classifier")
	parser.add_argument("--embedding", type=str, default="glove")
	parser.add_argument("--lookup", type=str, default="avg")
	parser.add_argument("--encoding", type=str, default="bilstm")
	parser.add_argument("--objective", type=str, default="cross_entropy")
	parser.add_argument("--metric", type=str, default='accuracy')


	parser.add_argument("--embed_size", type=int, default=1024)
	parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--optimizer", type=str, default = "adam")
	parser.add_argument("--num_epochs", type=int, default=20)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--l_rate", type=float, default=0.0001)
	parser.add_argument("--clip_threshold", type=float, default=10)
	parser.add_argument("--eval_interval", type=int, default=20)
	parser.add_argument("--patience", type=int, default=20)

	args = parser.parse_args()

	if args.cuda and torch.cuda.is_available():
		vars(args)['use_cuda'] = True
	else:
		vars(args)['use_cuda'] = False

	parameters = get_parameters(args)

	return parameters

def get_parameters(args):

	embedding = args.embedding
	lookup = args.lookup
	encoding = args.encoding
	objective = args.objective

	### MODEL SPECIFIC ARGUMENTS
	### TO	a config file
	vars(args)['K'] = 4

	## Objective : Classification ##
	vars(args)["output_size"] = len(DAMSL_TAGSET)

	##  vanilla ##
	vars(args)['lookup_kernel_size'] = 3
	vars(args)['lookup_stride'] = 1
	vars(args)['encoder_input_size'] = 300
	vars(args)['encoder_hidden_size'] =  100
	vars(args)['encoder_num_layers'] = 2

	## Hierarchical elmo encoder##
	vars(args)['elmo_dropout'] = 0.5
	vars(args)['elmo_input_size'] = 300
	vars(args)['elmo_hidden_size'] = 100
	vars(args)['elmo_num_layers'] = 2
	vars(args)['elmo_cell_size'] = 100
	vars(args)['elmo_requires_grad'] = True

	return args
