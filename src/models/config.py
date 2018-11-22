import argparse
from src.models.factory import MODEL_REGISTRY


def add_args(parser):
	## General Model parameters
	model = parser.add_argument_group("Model Parameters")

	model.add_argument("--model", type=str, default="dl_bow")
	model.add_argument("--multitask", action="store_true", default=False)
	model.add_argument("--mutlitask-procedure", type=str, default="multiplex")


	model.add_argument("--token-encoder", type=str, default="glove")
	model.add_argument("--fixed-token-encoder", action = "store_false", default=True)
	model.add_argument("--utterance-encoder", type=str, default="avg")
	model.add_argument("--fixed-utterance-encoder", action="store_false", default=True)
	model.add_argument("--conversation-encoder", type=str, default="bilstm")
	model.add_argument("--output-layer", type=str, default="mlp")
	model.add_argument("--output-weight", type=float, default=1.0)
	model.add_argument("--network", type=str, default="dl_classifier_network")

	## TODO: Each Objective and Metric also have wrapper classes and thier own config functions
	model.add_argument("--objective", type=str)
	model.add_argument("--objective-weights", type=str)
	## Multiple tasks can have multiple metrics
	model.add_argument("--metric", type=str, action="append")
	model.add_argument("--valid-metric", type=str)
	model.add_argument("--dropout", type=float, default=0.2)

	## Model Parameters for every model in registry
	for model_name in MODEL_REGISTRY:
		if hasattr(MODEL_REGISTRY[model_name], "add_args"):
			getattr(MODEL_REGISTRY[model_name], "add_args")(parser)



