import torch
import torch.optim as optim
import torch
from torch.autograd import Variable
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


class MultiTaskAverageMeter(object):
	def __init__(self, named_metrics):
		self.metrics = []
		for metric in named_metrics:
			d = {"name":metric , "metric":AverageMeter()}
			self.metrics.append(d)
		self.reset()

	def reset(self):
		for item in self.metrics:
			item["metric"].reset()

	def update(self, update_values):
		for idx, item in enumerate(self.metrics):
			item["metric"].update(update_values[idx][0], update_values[idx][1])

	def print_values(self):
		print_str = " "
		for idx, item in enumerate(self.metrics):
			print_str += item["name"] + ": " + "%.4f".format(item["metric"].avg) + " ;"
		return print_str

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