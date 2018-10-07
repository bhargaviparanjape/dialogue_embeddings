import argparse,logging

def str2bool(v):
	return v.lower() in ('yes', 'true', 't', '1', 'y')

def add_args(parser):
	parser.register('type', 'bool', str2bool)

	# Runtime environment
	runtime = parser.add_argument_group('Environment')
	runtime.add_argument("--cuda", action="store_true", default=True)
	runtime.add_argument("--seed", type=int, default=0)
	runtime.add_argument("--truncate-dataset", action="store_true", default=False)
	runtime.add_argument("--log_-level", type=str, default=logging.INFO)
	runtime.add_argument('--run-mode', type=str, default="train", help="Run mode: {train, test}")

	# Files
	files = parser.add_argument_group('Filesystem')
	files.add_argument('--dataset', type=str, action="append")
	files.add_argument('--dataset-path', type=str, action="append")
	files.add_argument("--log-file", type=str, default=None)
	files.add_argument("--pretrained-embedding-path", type=str, default=None)
	files.add_argument("--model-path", type=str, default=None)
	files.add_argument('--model-dir', type=str, default=None)
	files.add_argument("--pretrained-model-path", type=str, default=None)

	# Saving + loading
	save_load = parser.add_argument_group('Saving/Loading')
	save_load.add_argument('--checkpoint', type='bool', default=False,
						   help='Save model + optimizer state after each epoch')
	save_load.add_argument('--pretrained-model', type=str, default='',
						   help='Path to a pretrained model to warm-start with')




def set_defaults(args):
	raise NotImplementedError






