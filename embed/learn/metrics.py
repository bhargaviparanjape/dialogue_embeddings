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
		self.total += mask.sum().data.numpy()
		predicted = input[0]*mask
		correct = input[1]*mask
		self.correct += ((predicted == correct).long()*mask).sum().numpy()

	def reset(self):
		self.correct = 0
		self.total = 0


@RegisterMetric('bidi_accuracy')
class BidirectionalAccuracy(Accuracy):
	def __init__(self, args):
		super(BidirectionalAccuracy, self).__init__(args)

	def update_metric(self, batch_size, *input):
		mask = input[2].view(-1, 1).squeeze(1)
		self.total += mask.sum().data.numpy()
		next_predicted = input[0][0] * mask
		prev_predicted = input[0][1] * mask
		next_correct = input[1][0] * mask
		prev_correct = input[1][1] * mask
		self.correct += (((next_predicted == next_correct) == (prev_predicted == prev_correct)).long()*mask).sum().numpy()


@RegisterMetric('bow')
class BagOfWordMetric():
	def __init__(self, args):
		self.args = args

	def update_metric(self, batch_size, *input):
		mask = input[2]
		predicted = input[0]
		gold_ids = input[1]
		self.correct += ((predicted[0] == gold_ids[0]).float() * mask).sum().numpy()


	def compute_metric(self):
		## how many match exactly / how many total
		if self.total == 0:
			self.value = 0
			return 0
		self.value = float(self.correct)/self.total
		return self.value

	def reset(self):
		self.correct = 0
		self.total = 0

@RegisterMetric('label_accuracy')
class LabelAccuracy(BidirectionalAccuracy):
	def __init__(self, args):
		super(LabelAccuracy, self).__init__(args)
		self.correct_labels = 0
		self.total_labels = 0

	def update_metric(self, batch_size, *input):
		super(LabelAccuracy, self).update_metric(batch_size, *input)
		mask = input[2].view(-1, 1).squeeze(1)
		self.total_labels += mask.sum().data.numpy()
		labels_predicted = input[0][2] * mask
		labels_correct = input[1][2] * mask
		self.correct_labels += ((labels_predicted == labels_correct).long()*mask).sum().numpy()

	def compute_metric(self):
		utterance_prediction_accuracy = super(LabelAccuracy, self).compute_metric()
		if self.total_labels == 0:
			self.label_accuracy = 0
			return 0
		self.label_accuracy = float(self.correct)/self.total
		return self.label_accuracy