import sys,logging,argparse
from os.path import dirname, realpath
import numpy as np
import torch
import random
sys.path.append(dirname(dirname(realpath(__file__))))
from src.utils import global_parameters
from src.learn import config as train_config
from src.models import config as model_config
from src.dataloaders import factory as dataloader_factory
from allennlp.modules.elmo import Elmo, batch_to_ids
import json


def average_embeddings(embeddings, mask):
	output = (embeddings*mask.unsqueeze(2)).sum(1) / mask.sum(1).unsqueeze(1)
	output[output != output] = 0
	return output.data.numpy().tolist()


def get_pretrained_embeddings(args, dataset):
	options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
	weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
	ee = Elmo(options_file, weight_file, requires_grad=False, num_output_representations=1, dropout=args.dropout)
	with open(args.output_path , "w+") as output_path:
		for sub_dataset in [dataset.train_dataset, dataset.valid_dataset, dataset.test_dataset]:
			for conversation in sub_dataset:
				conversation_dict = {}
				conversation_id = conversation.id
				print(conversation_id)
				conversation_dict["id"] = conversation_id
				utterances = [u.tokens for u in conversation.utterances]
				character_ids = batch_to_ids(utterances)
				embeddings = torch.FloatTensor(character_ids.shape[0], character_ids.shape[1], 1024)
				mask = torch.Tensor(character_ids.shape[0], character_ids.shape[1])
				for i in range(0, character_ids.shape[0], 10):
					dict = ee(character_ids[i:i + 10].unsqueeze(0))
					embeddings[i:i + 10] = dict['elmo_representations'][0]
					mask[i:i + 10] = dict['mask']
				if args.utterance_encoder == "avg":
					conversation_embeddings = average_embeddings(embeddings, mask)
					conversation_dict["embeddings"] = conversation_embeddings
				output_path.write(json.dumps(conversation_dict)+"\n")



if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		'Preproces Data',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	global_parameters.add_args(parser)
	train_config.add_args(parser)
	model_config.add_args(parser)
	args = parser.parse_args()
	global_parameters.add_config(args, sys.argv[1])

	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger(__name__)

	dataset = dataloader_factory.get_dataset(args, logger)
	get_pretrained_embeddings(args, dataset)