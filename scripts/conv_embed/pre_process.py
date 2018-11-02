import torch
import torch.multiprocessing as mp
try:
	mp.set_start_method('spawn')
except:
	pass
#from allennlp.modules.elmo import batch_to_ids

def average_embeddings(embeddings, mask):
		output = (embeddings*mask.unsqueeze(2)).sum(1) / mask.sum(1).unsqueeze(1)
		output[output != output] = 0
		return output.data.cpu().numpy().tolist()


def process_conversation(data):
	gpu_no = data[0]
	func = data[1]
	conversation_ids = data[2]
	ee = data[3]
	batch_data = []
	snippet_size = 40
	for conversation in conversation_ids:
		conversation_dict = {}
		conversation_id = conversation.id
		print(conversation_id)
		conversation_dict["id"] = conversation_id
		utterances = [u.tokens for u in conversation.utterances]
		character_ids = func(utterances)
		embeddings = torch.FloatTensor(character_ids.shape[0], character_ids.shape[1], 1024).cuda(gpu_no)
		mask = torch.Tensor(character_ids.shape[0], character_ids.shape[1]).cuda(gpu_no)
		for i in range(0, character_ids.shape[0], snippet_size):
			dict = ee(character_ids[i:i + snippet_size].unsqueeze(0))
			embeddings[i:i + snippet_size] = dict['elmo_representations'][0]
			mask[i:i + snippet_size] = dict['mask']
		conversation_embeddings = average_embeddings(embeddings, mask)
		conversation_dict["embeddings"] = conversation_embeddings
		batch_data.append(conversation_dict)
	return batch_data

	
if __name__ == "__main__":
	#mp.set_start_method('spawn')
	import sys,logging,argparse,pdb
	from os.path import dirname, realpath
	import numpy as np
	import torch
	import random
	sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
	from src.utils import global_parameters
	from src.learn import config as train_config
	from src.models import config as model_config
	from src.dataloaders import factory as dataloader_factory
	from allennlp.modules.elmo import Elmo
	from allennlp.modules.elmo import batch_to_ids
	import json
	from src.utils import global_parameters

	def get_pretrained_embeddings(args, dataset, ee):
		job_pool = mp.Pool(args.data_workers, maxtasksperchild=1)
		job_data = []
		num_gpus = torch.cuda.device_count()
		assigned_gpu = 0
		for sub_dataset in [dataset.train_dataset, dataset.valid_dataset, dataset.test_dataset]:
			## Assign Available GPU in Round robin order
			for conversation_set in range(0, len(sub_dataset), 10):
				job_data.append([assigned_gpu, batch_to_ids, sub_dataset[conversation_set:conversation_set+10], ee])
				assigned_gpu += 1
				if assigned_gpu == num_gpus:
					assigned_gpu = 0

		## CPU MULTIPROCESSING
		with open(args.output_path, "w+") as output_path:
			for result in job_pool.imap(process_conversation, job_data):
				for r in result:
					output_path.write(json.dumps(r) + "\n")

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

	options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
	weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
	ee = Elmo(options_file, weight_file, requires_grad=False, num_output_representations=1, dropout=args.dropout)
	ee.share_memory()
	get_pretrained_embeddings(args, dataset, ee)
