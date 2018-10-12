import torch
import torch.optim as optim
import torch
from torch.autograd import Variable
import numpy as np
import time

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

def variable(v, arg_use_cuda=True, volatile=False):
	if torch.cuda.is_available() and arg_use_cuda:
		return Variable(v, volatile=volatile).cuda()
	return Variable(v, volatile=volatile)


def select_optimizer(args, parameters):
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(parameters, args.l_rate,
								   momentum=args.momentum,
								   weight_decay=args.weight_decay)
	elif args.optimizer == 'adamax':
		optimizer = optim.Adamax(parameters,
									  weight_decay=args.weight_decay)
	elif args.optimizer == "adam":
		optimizer = optim.Adam(parameters, lr=args.l_rate)
	else:
		raise RuntimeError('Unsupported optimizer: %s' %
						   args.optimizer)

	return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageCounter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, total):
		self.val = val
		self.sum += val
		self.count += total
		self.avg = self.sum / self.count


class MultiTaskAverageCounter(object):
	def __init__(self, named_metrics):
		self.metrics = []
		for metric in named_metrics:
			d = {"name":metric , "metric":AverageCounter()}
			self.metrics.append(d)
		self.reset()

	def reset(self):
		for item in self.metrics:
			item["metric"].reset()

	def update(self, update_values):
		for idx, item in enumerate(self.metrics):
			item["metric"].update(update_values[item["name"]][0], update_values[item["name"]][1])

	def print_values(self):
		print_str = " "
		for idx, item in enumerate(self.metrics):
			print_str += item["name"] + ": " + "%.4f" % (item["metric"].avg) + " ;"
		return print_str

	def average(self):
		metric_values  = []
		for idx, item in enumerate(self.metrics):
			metric_values.append(item["metric"].avg)
		return np.mean(metric_values)

	def validation_metric(self, valid_metric):
		if len(valid_metric) == 0:
			return self.average()
		else:
			metric_values = []
			for idx, item in enumerate(self.metrics):
				if item["name"] in valid_metric:
					metric_values.append(item["metric"].avg)
			return np.mean(metric_values)
		return None

class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total

def pad_seq(seq, max_len, pad_token=0):
    seq += [pad_token for i in range(max_len - len(seq))]
    return seq