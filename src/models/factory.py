MODEL_REGISTRY = {}
NO_MODEL_ERR = "Model {} not in MODEL_REGISTRY! Available models are {}"

def RegisterModel(model_name):
	"""Registers a model."""

	def decorator(f):
		MODEL_REGISTRY[model_name] = f
		return f

	return decorator

def get_embeddings(embedding_layer_name, args):
	if embedding_layer_name not in MODEL_REGISTRY:
		raise Exception(
			NO_MODEL_ERR.format(embedding_layer_name, MODEL_REGISTRY.keys()))
	if embedding_layer_name in MODEL_REGISTRY:
		model = MODEL_REGISTRY[embedding_layer_name](args)
	return model

def get_model_collection_by_name(model_name, args, **kwargs):
	pass

def get_model_by_name(model_name, args, **kwargs):
	if model_name not in MODEL_REGISTRY:
		raise Exception(
			NO_MODEL_ERR.format(model_name, MODEL_REGISTRY.keys()))

	if model_name in MODEL_REGISTRY:
		model = MODEL_REGISTRY[model_name](args, **kwargs)

	return model

def get_model(args, **kwargs):
	## Split into components
	main_model = get_model_by_name(args.model, args, **kwargs)
	return main_model
