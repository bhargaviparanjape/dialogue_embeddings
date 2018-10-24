import tqdm,os,sys,logging
import torch

from src.dataloaders import  factory as dataloader_factory
from src.models import factory as model_factory
from src.utils.utility_functions import AverageMeter,Timer, MultiTaskAverageCounter
from random import shuffle
logger = logging.getLogger(__name__)


def train_epochs_multiplex(args, dataset, model):
	pass

def train_epochs(args, dataset, model):

	#train_batches, validation_batches, test_batches = dataloader_factory.get_batches(args, dataset)
	train_dataloader, validation_dataloader, test_dataloader = dataloader_factory.get_dataloader(args, dataset)

	start_epoch = 0
	stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0}
	for epoch in range(start_epoch, args.num_epochs):
		stats['epoch'] = epoch
		train_loss = AverageMeter()
		epoch_time = Timer()

		# Run 1 epoch
		for iteration, batch in enumerate(train_dataloader):
			train_loss.update(*model.update(batch))

			if (iteration + 1) % args.eval_interval == 0:
				logger.info('train: Epoch = %d | iter = %d/%d | ' %
						(epoch, iteration, len(train_dataloader)) +
						'loss = %.2f | elapsed time = %.2f (s)' %
						(train_loss.avg, stats['timer'].time()))
				train_loss.reset()

		logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
						(epoch, epoch_time.time()))

		# Checkpoint model after every epoch
		if args.checkpoint:
			model.checkpoint(args.model_file + ".checkpoint", epoch + 1)


		# Validate Partial Train data
		validate(args, train_dataloader, model, stats, mode = "train")

		# Validate on Development data
		# TODO: Make this more generic and abstract out model file
		result = validate(args, validation_dataloader, model, stats, mode = "dev")

		if result > stats['best_valid']:
			logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
						(" ".join(args.metric), result,
						 stats['epoch'], model.updates))
			model.save()
			stats['best_valid'] = result

		# Recreate training batches using shuffle
		logger.info("Creating train batches for epoch {0}".format(epoch+1))
		train_batches,_,_ = dataloader_factory.get_batches(args, dataset)
		shuffle(train_batches)


def validate(args, dataloader, model, stats, mode = "dev"):

	eval_time = Timer()
	metrics = MultiTaskAverageCounter(args.metric)

	examples = 0
	for iteration, batch in enumerate(dataloader):
		examples += len(batch)
		pred, mask  = model.predict(batch)
		target,_ = model.target(batch)

		update_metrics = model.evaluate_metrics(pred, target, mask, mode)
		metrics.update(update_values=update_metrics)

		if mode == 'train' and examples >= 1e4:
			break

	logger.info('%s valid : Epoch = %d | metrics = %s | ' %
				(mode, stats['epoch'], metrics.print_values()) +
				'examples = %d | ' %
				(examples) +
				'valid time = %.2f (s)' % eval_time.time())

	## accumulate score
	# The metrics that will be considered in the validation history and used for patience computation
	result = metrics.validation_metric(args.valid_metric)
	return result


