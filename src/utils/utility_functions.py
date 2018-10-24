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
	def __init__(self, models):
		self.metrics = {}
		for model in models:
			self.metrics[model["model"]] = []
			for metric in model["metric"]:
				d = {"name":metric , "metric":AverageCounter()}
				self.metrics[model["model"]].append(d)
		self.reset()

	def reset(self):
		for item in self.metrics:
			for i in range(len(self.metrics[item])):
				(self.metrics[item][i]["metric"]).reset()

	def update(self, update_values):
		for key in self.metrics:
			metric_update_values = update_values[key]
			for i in range(len(self.metrics[key])):
				metric_name = self.metrics[key][i]["name"]
				(self.metrics[key][i]["metric"]).update(metric_update_values[metric_name][0], metric_update_values[metric_name][1])

	def print_values(self):
		print_str = " "
		for key in self.metrics:
			metric_by_model = self.metrics[key]
			print_str += "%s{ "%key
			for idx, item in enumerate(metric_by_model):
				print_str += item["name"] + ": " + "%.4f" % (item["metric"].avg) + " ;"
			print_str += " },"
		return print_str

	def average(self):
		metric_values  = []
		for key in self.metrics:
			for idx, item in enumerate(self.metrics[key]):
				metric_values.append(item["metric"].avg)
		return np.mean(metric_values)

	def validation_metric(self, valid_metric):
		## TODO: Add support for validation metrics of a single task only; currently an average is considered
		valid_metric_dict = {k:v for (k,v) in valid_metric}
		if len(valid_metric_dict) == 0:
			return self.average()
		else:
			metric_values = []
			# for key in self.metrics:
			# 	for idx, item in enumerate(self.metrics[key]):
			# 		if item["name"] in valid_metric:
			# 			metric_values.append(item["metric"].avg)
			for key, value in valid_metric_dict.items():
				task_metrics = self.metrics[key]
				for idx, item in enumerate(task_metrics):
					if item["name"] == value:
						metric_values.append(item["metric"].avg)
						break
			return np.mean(metric_values)

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