from src.dataloaders.AbstractDataset import AbstractDataset

NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"
NO_BACTHER_ERR = "Bacthing function {} not in BACTHER_REGISTRY! Available batching functions are {}"

DATASET_REGISTRY = {}
BATCHER_REGISTRY = {}

def RegisterDataset(dataset_name):
	"""Registers a dataset."""

	def decorator(f):
		DATASET_REGISTRY[dataset_name] = f
		return f

	return decorator

def RegisterBatcher(batching_function_name):
	"""Registers a dataset."""

	def decorator(f):
		BATCHER_REGISTRY[batching_function_name] = f
		return f

	return decorator


def get_dataset(args, logger):
	## TODO: These are already lists now
	datasets = args.dataset
	dataset_paths = args.dataset_path
	aggregated_dataset = AbstractDataset()
	for idx, dataset in enumerate(datasets):
		if dataset not in DATASET_REGISTRY:
			raise Exception(
				NO_DATASET_ERR.format(dataset, DATASET_REGISTRY.keys()))

		if dataset in DATASET_REGISTRY:
			loaded_dataset = DATASET_REGISTRY[dataset](args, dataset_paths[idx])
			aggregated_dataset += loaded_dataset


	return aggregated_dataset

def get_batches(args, dataset):
	if args.batch_function not in BATCHER_REGISTRY:
		raise Exception(
			NO_BACTHER_ERR.format(args.batch_function, BATCHER_REGISTRY.keys()))

	if args.batch_function in BATCHER_REGISTRY:
		batcher = BATCHER_REGISTRY[args.batch_function](args)
		batches = batcher.get_batches(args, dataset)

	return batches


def set_dataset_arguments(args, dataset):
	if args.model == "da_classifier":
		vars(args)["output_size"] = dataset.label_set_size
	elif args.model == "dl_bow" or args.model == "dl_bow2":
		vars(args)["output_size"] = len(dataset.vocabulary.vocabulary)
	elif args.model == "da_bow":
		vars(args)["output_size"] = [len(dataset.vocabulary.vocabulary), dataset.label_set_size]