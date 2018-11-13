import argparse

def add_args(parser):

	## Training Regime
	training = parser.add_argument_group("Training")

	training.add_argument("--batch-function", type=str, default="conversation_length")
	training.add_argument("--dataloader", type=str, default="conversation_snippets")
	training.add_argument("--num-epochs", type=int, default=20)
	training.add_argument("--optimizer", type=str, default="adam")
	training.add_argument("--batch-size", type=int, default=8)
	training.add_argument("--eval-batch-size", type=int, default=32)
	training.add_argument("--clip-threshold", type=float, default=10)
	training.add_argument("--eval-interval", type=int, default=20)
	training.add_argument("--save-interval", type=int, default=1000)
	training.add_argument("--patience", type=int, default=30)
	training.add_argument("--conversation-size", type=int, default=32)


	## Optimizer parameters for sgd,adam, adamax
	optimizer = parser.add_argument_group("Optimizer Parameters")
	optimizer.add_argument("--l-rate", type=float, default=0.0001)
	optimizer.add_argument("--momentum" , type=float, default=None)
	optimizer.add_argument("--weight-decay", type=float, default=None)

