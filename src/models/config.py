import argparse
from src.models.factory import MODEL_REGISTRY


def add_args(parser):
	## General Model parameters
	model = parser.add_argument_group("Model Parameters")

	model.add_argument("--model", type=str, default="dialogue_classifier")

	model.add_argument("--token-encoder", type=str, default="glove")
	model.add_argument("--fixed-token-encoder", action = "store_false", default=True)
	model.add_argument("--utterance-encoder", type=str, default="avg")
	model.add_argument("--fixed-utterance-encoder", action="store_false", default=True)
	model.add_argument("--conversation-encoder", type=str, default="bilstm")
	model.add_argument("--output-layer", type=str, default=[], action = "append")
	model.add_argument("--output-weights", type=float, default=[], action = "append")
	model.add_argument("--network", type=str, default="dl_classifier_network")

	## TODO: Each Objective and Metric also have wrapper classes and thier own config functions
	model.add_argument("--objective", type=str, action="append")
	model.add_argument("--objective-weights", type=str, action="append")
	model.add_argument("--metric", type=str, action="append")
	model.add_argument("--dropout", type=float, default=0.2)

	## Model Parameters for every model in registry
	for model_name in MODEL_REGISTRY:
		if hasattr(MODEL_REGISTRY[model_name], "add_args"):
			getattr(MODEL_REGISTRY[model_name], "add_args")(parser)



