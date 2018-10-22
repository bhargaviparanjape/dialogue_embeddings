## Import All Datasets and print thier metadata

import os,sys,argparse,pdb,numpy as np,logging
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from src.utils import global_parameters
from src.dataloaders import factory as data_factroy

logger = logging.getLogger()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		'Dataset Statistics',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	# General Arguments + File Paths + Dataset Paths
	global_parameters.add_args(parser)
	args = parser.parse_args()

	logger.setLevel(logging.DEBUG)
	fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
	console = logging.StreamHandler()
	console.setFormatter(fmt)
	logger.addHandler(console)

	datasets = data_factroy.get_dataset_list(args, logger)

	for dataset in datasets:
		logger.info("Name: %s" % dataset.name)
		data = dataset.get_full_dataset()
		logger.info("Length: %d" % dataset.total_length)
		avg_transcript_length = 0
		max_transcript_length = -1
		avg_utterance_length = 0
		max_utterance_length = -1
		for transcription in data:
			avg_transcript_length += len(transcription.utterances)
			max_transcript_length = max(max_transcript_length, len(transcription.utterances))
			if len(transcription.utterances) == 0:
				continue
			avg_utterance_length += sum([u.length for u in transcription.utterances])/len(transcription.utterances)
			max_utterance_length = max(max_utterance_length, max([u.length for u in transcription.utterances]))
		avg_utterance_length = float(avg_utterance_length)/dataset.total_length
		avg_transcript_length = float(avg_transcript_length)/dataset.total_length
		logger.info("Avg. T length : %.4f" % avg_transcript_length)
		logger.info("Max T length : %.4f" % max_transcript_length)
		logger.info("Avg. U length : %.4f" % avg_utterance_length)
		logger.info("Max U length : %.4f" % max_utterance_length)

