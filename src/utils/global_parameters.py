import argparse,logging
import json,os,sys

MAX_VOCAB_LENGTH = 5000

def str2bool(v):
	return v.lower() in ('yes', 'true', 't', '1', 'y')

def add_args(parser):
	parser.register('type', 'bool', str2bool)

	# Runtime environment
	runtime = parser.add_argument_group('Environment')
	runtime.add_argument("--cuda", action="store_true", default=True)
	runtime.add_argument("--parallel", action="store_true", default=False)
	runtime.add_argument("--seed", type=int, default=0)
	runtime.add_argument("--data_workers", type=int, default=1)
	runtime.add_argument("--truncate-dataset", action="store_true", default=False)
	runtime.add_argument("--limit-vocabulary", action="store_true", default=False)
	runtime.add_argument("--log-level", type=str, default=logging.INFO)
	runtime.add_argument('--run-mode', type=str, default="train", help="Run mode: {train, test}")

	# Files
	files = parser.add_argument_group('Filesystem')
	files.add_argument('config-file', type=str, default="bow.json")
	files.add_argument('--dataset', type=str, default="swda")
	files.add_argument('--dataset-path', type=str, default="datasets/swda/swda")
	files.add_argument("--log-file", type=str, default=None)
	files.add_argument("--pretrained-embedding-path", type=str, default=None)
	files.add_argument("--model-path", type=str, default=None)
	files.add_argument('--model-dir', type=str, default=None)
	files.add_argument("--pretrained-model-path", type=str, default=None)
	files.add_argument("--pretrained-dialogue-embed-path", type=str, default=None)
	files.add_argument("--output-path", type=str, default=None)

	# Saving + loading
	save_load = parser.add_argument_group('Saving/Loading')
	save_load.add_argument('--checkpoint', type='bool', default=False,
						   help='Save model + optimizer state after each epoch')
	save_load.add_argument('--pretrained-model', type=str, default='',
						   help='Path to a pretrained model to warm-start with')


def add_config(args, config_file):
	config_arguments = json.load(open(config_file))
	dataset_arguments = config_arguments.get("datasets", None)
	vars(args)["datasets"] = dataset_arguments
	model_arguments = config_arguments.get("models", None)
	vars(args)["models"] = model_arguments

def set_defaults(args):
	raise NotImplementedError






