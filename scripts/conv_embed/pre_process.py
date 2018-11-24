import torch
import torch.multiprocessing as mp
from src.utils.utility_functions import pad_seq
import h5py
import pickle
import tqdm
try:
	mp.set_start_method('spawn')
except:
	pass
#from allennlp.modules.elmo import batch_to_ids

def average_embeddings(embeddings, mask):
	output = (embeddings*mask.unsqueeze(2)).sum(1) / mask.sum(1).unsqueeze(1)
	output[output != output] = 0
	return output.data.cpu().numpy()

def weighted_average_embeddings(embeddings, mask, idfs):
	output = (embeddings * mask.unsqueeze(2)*idfs.unsqueeze(2)).sum(1) / mask.sum(1).unsqueeze(1)
	output[output != output] = 0
	return output.data.cpu().numpy()


def process_conversation(data):
	gpu_no = data[0]
	func = data[1]
	conversation_ids = data[2]
	ee = data[3]
	idfs = data[4]
	batch_data = []
	snippet_size = 40
	for enum_, conversation in enumerate(conversation_ids):
		conversation_id = conversation.id
		utterances = [u.tokens for u in conversation.utterances]
		character_ids = func(utterances)
		if torch.cuda.is_available():
			embeddings = torch.FloatTensor(character_ids.shape[0], character_ids.shape[1], 1024).cuda(gpu_no)
			mask = torch.Tensor(character_ids.shape[0], character_ids.shape[1]).cuda(gpu_no)
			idf_tensor = torch.FloatTensor(idfs[enum_]).cuda(gpu_no)
			hidden_layers = torch.FloatTensor(character_ids.shape[0], 1024).cuda(gpu_no)
		else:
			embeddings = torch.FloatTensor(character_ids.shape[0], character_ids.shape[1], 1024)
			mask = torch.Tensor(character_ids.shape[0], character_ids.shape[1])
			hidden_layers = torch.FloatTensor(character_ids.shape[0], 1024)
			idf_tensor = torch.FloatTensor(idfs[enum_])
		for i in range(0, character_ids.shape[0], snippet_size):
			dict = ee(character_ids[i:i + snippet_size].unsqueeze(0))
			embeddings[i:i + snippet_size] = dict['elmo_representations'][0]
			mask[i:i + snippet_size] = dict['mask']
			# check for last layer;
			hidden_layers[i:i + snippet_size] = \
				ee._elmo_lstm._elmo_lstm._states[0][1][:hidden_layers[i:i + snippet_size].size(0)]
		conversation_embeddings = average_embeddings(embeddings, mask)
		# Variant 1 : Try weighing the different words with idf so that information about hte msot important words is retained
		weighted_conversation_embeddings = weighted_average_embeddings(embeddings, mask, idf_tensor)
		hidden_layers = hidden_layers.data.cpu().numpy()
		# Variant 2: Instead of average, provide the hidden representation of the elmo embeddings
		batch_data.append((conversation_id, conversation_embeddings, weighted_conversation_embeddings, hidden_layers))
		ee._elmo_lstm._elmo_lstm._states = None
	return batch_data

	
if __name__ == "__main__":
	import sys,logging,argparse, pdb
	from os.path import dirname, realpath
	import numpy as np
	import torch
	sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
	from src.utils import global_parameters
	from src.learn import config as train_config
	from src.models import config as model_config
	from src.dataloaders import factory as dataloader_factory
	from allennlp.modules.elmo import Elmo
	from allennlp.modules.elmo import batch_to_ids
	import json
	from src.utils import global_parameters

	def get_ids(sentence_list):
		max_len = max([u.length for u in sentence_list])
		ids = [pad_seq(dataset.vocabulary.get_indices(u.tokens), max_len) for u in sentence_list]
		return ids

	def get_pretrained_embeddings(args, dataset, ee):
		job_pool = mp.Pool(args.data_workers, maxtasksperchild=1)
		job_data = []
		num_gpus = torch.cuda.device_count()
		assigned_gpu = 0
		for sub_dataset in [dataset.train_dataset, dataset.valid_dataset, dataset.test_dataset]:
			## Assign Available GPU in Round robin order
			for conversation_set in range(0, len(sub_dataset), 10):
				batch_ids = [get_ids(conversation.utterances) for conversation in sub_dataset[conversation_set:conversation_set+10]]
				batch_idfs = [dataset.vocabulary.inverse_utterance_frequency[np.array(ids)] for ids in batch_ids]
				job_data.append([assigned_gpu, batch_to_ids, sub_dataset[conversation_set:conversation_set+10], ee, batch_idfs])
				assigned_gpu += 1
				if assigned_gpu == num_gpus:
					assigned_gpu = 0

		h_file = h5py.File(args.output_hdf5_path, "w")
		dt = h5py.special_dtype(vlen=np.dtype('float64'))
		feature_length = 1024
		pkl_file = args.output_pkl_path
		conversation_id_map = {}
		num_conversations = dataset.total_length

		average_elmo_features = h_file.create_dataset(
			'average_elmo', (num_conversations, feature_length), dtype=dt)
		weighted_elmo_features = h_file.create_dataset(
			'weighted_elmo', (num_conversations, feature_length), dtype=dt)
		final_elmo_features = h_file.create_dataset(
			'final_elmo', (num_conversations, feature_length), dtype=dt)

		# CPU only checking
		process_conversation(job_data[0])

		conversation_processed = 0
		## TQDM on this
		for result in tqdm.tqdm(job_pool.imap(process_conversation, job_data)):
			for r in result:
				# dump conversation_ids mapped to conversation_processed
				# dump conversation_processed into h5py
				conversation_id, average_elmo, weighted_elmo, final_elmo = r
				conversation_id_map[conversation_id] = conversation_processed
				average_elmo_features[conversation_processed] = average_elmo.transpose() #reshape
				weighted_elmo_features[conversation_processed] = weighted_elmo.transpose()
				final_elmo_features[conversation_processed] = final_elmo.transpose()
				conversation_processed += 1


		h_file.close()
		pickle.dump(conversation_id_map, open(pkl_file, 'wb'))


	parser = argparse.ArgumentParser(
			'Preproces Data',
			formatter_class=argparse.ArgumentDefaultsHelpFormatter
		)
	parser.add_argument("--output-hdf5-path", type=str, default=None)
	parser.add_argument("--output-pkl-path", type=str, default=None)
	global_parameters.add_args(parser)
	train_config.add_args(parser)
	model_config.add_args(parser)
	parser.add_argument('--weight-embeddings', action="store_true", default=False)
	args = parser.parse_args()
	global_parameters.add_config(args, sys.argv[1])

	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger(__name__)

	dataset = dataloader_factory.get_dataset(args, logger)
	dataset.vocabulary.compute_inverse_frequency(dataset.utterance_length)

	options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
	weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
	ee = Elmo(options_file, weight_file, requires_grad=False, num_output_representations=1, dropout=args.dropout)
	ee.share_memory()
	get_pretrained_embeddings(args, dataset, ee)
