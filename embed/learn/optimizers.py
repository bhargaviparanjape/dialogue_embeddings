from torch.optim import Adam, Adadelta, SGD, Adagrad

NO_OPTIMIZER_ERR = "Model {} not in OPTIMIZER_REGISTRY! Available optimizers are {}"
OPTIMIZER_REGISTRY = {}

def RegisterOptimzer(optimizer_name):
	"""Registers a optimizer."""
	def decorator(f):
		OPTIMIZER_REGISTRY[optimizer_name] = f
		return f
	return decorator


def get_optimizer(args, model):
	if args.optimizer not in OPTIMIZER_REGISTRY:
		raise Exception(
			NO_OPTIMIZER_ERR.format(args.optimizer, OPTIMIZER_REGISTRY.keys()))

	if args.optimizer in OPTIMIZER_REGISTRY:
		optimizer = OPTIMIZER_REGISTRY[args.optimizer](args, model)

	return optimizer


@RegisterOptimzer('adam')
class Adam_():
	def __init__(self, args, model):
		self.optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.l_rate)

@RegisterOptimzer('adagrad')
class Adagrad_():
	def __init__(self, args, model):
		self.optimizer = Adadelta(model.parameters(), lr=args.l_rate)
