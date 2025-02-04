import numpy as np
from sklearn.metrics import f1_score,cohen_kappa_score

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
		self.multitask = False
		self.correct_collect = []
		self.predict_collect = []

	def compute_f1(self):
		return cohen_kappa_score(self.correct_collect, self.predict_collect)  

	def compute_metric(self):
		if self.total == 0:
			self.value = 0
			return 0
		self.value = float(self.correct)/self.total
		return self.value

	def update_metric(self, batch_size, *input):
		mask = input[2].view(-1,1).squeeze(1)
		self.total += mask.sum().data.numpy()
		predicted = input[0][0]*mask
		correct = input[1][0]*mask
		self.correct += ((predicted == correct).long()*mask).sum().numpy()
		self.correct_collect += correct.long().data.numpy().tolist()
		self.predict_collect += predicted.long().data.numpy().tolist()

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
		self.correct = 0
		self.total = 0
		self.multitask = False

	def update_metric(self, batch_size, *input):
		mask = input[2]
		## mask both input[0][0] and inpit[1][0] and then check if the sets have intersection
		masked_predicted = input[0][0]*mask.long()
		masked_gold = input[1][0]*mask.long()
		for i in range(input[0][0].shape[0]):
			predicted_set = set([j for j in masked_predicted.numpy()[i].tolist() if j != 0])
			gold_set =  set([j for j in masked_gold.numpy()[i].tolist() if j != 0])
			self.correct += len(predicted_set & gold_set)
		## label accuracy
		self.total += mask.sum().data.numpy()

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
		self.multitask = True

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
		self.label_accuracy = float(self.correct_labels)/self.total_labels
		return self.label_accuracy

@RegisterMetric('bow_label_accuracy')
class LabelAccuracy(BagOfWordMetric):
	def __init__(self, args):
		super(LabelAccuracy, self).__init__(args)
		self.correct_labels = 0
		self.total_labels = 0
		self.multitask = True

	def update_metric(self, batch_size, *input):
		super(LabelAccuracy, self).update_metric(batch_size, *input)
		mask_sum = input[2].sum(1)
		mask_sum[mask_sum!= 0] = 1
		mask = mask_sum.view(-1, 1).squeeze(1)
		self.total_labels += mask.sum().data.numpy()
		labels_predicted = input[0][1] * mask.long()
		labels_correct = input[1][1] * mask.long()
		self.correct_labels += ((labels_predicted == labels_correct).long()*mask.long()).sum().numpy()

	def compute_metric(self):
		utterance_prediction_accuracy = super(LabelAccuracy, self).compute_metric()
		if self.total_labels == 0:
			self.label_accuracy = 0
			return 0
		self.label_accuracy = float(self.correct_labels)/self.total_labels
		return self.label_accuracy
