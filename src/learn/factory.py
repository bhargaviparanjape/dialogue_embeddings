OBJECTIVE_REGISTRY = {}
NO_OBJECTIVEL_ERR = "Objective {} not in OBJECTIVE_REGISTRY! Available objectives are {}"

def RegisterObjective(name):
	"""Registers a model."""

	def decorator(f):
		OBJECTIVE_REGISTRY[name] = f
		return f

	return decorator

def get_objective(args, logger=None):

	## Split into components
	objective_object = OBJECTIVE_REGISTRY[args.objective](args, logger)
	return objective_object