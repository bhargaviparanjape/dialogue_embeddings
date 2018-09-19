NO_METRIC_ERR = "Model {} not in METRIC_REGISTRY! Available metrics are {}"
METRIC_REGISTRY = {}

def RegisterMetric(metric_name):
	"""Registers a metric."""
	def decorator(f):
		METRIC_REGISTRY[metric_name] = f
		return f
	return decorator


def get_metric(args):
	if args.metric not in METRIC_REGISTRY:
		raise Exception(
			NO_METRIC_ERR.format(args.metric, METRIC_REGISTRY.keys()))

	if args.metric in METRIC_REGISTRY:
		metric = METRIC_REGISTRY[args.metric](args)

	return metric


@RegisterMetric('accuracy')
class Accuracy():
	def __init__(self, args):
		self.args = args
		self.correct = 0
		self.total = 0

	def compute_metric(self):
		if self.total == 0:
			self.value = 0
			return 0
		self.value = float(self.correct)/self.total
		return float(self.correct)/self.total

	def update_metric(self, batch_size, *input):
		mask = input[2].view(-1,1).squeeze(1)
		self.total += mask.sum()
		predicted = input[0]*mask
		correct = input[1]*mask
		self.correct += ((predicted == correct).numpy()*mask).sum()


