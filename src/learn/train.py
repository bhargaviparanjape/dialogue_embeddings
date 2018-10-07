import tqdm,os,sys
import torch

from src.dataloaders import  factory as dataloader_factory
from src.models import factory as model_factory
from src.utils.utility_functions import AverageMeter,Timer, MultiTaskAverageMeter

def train_epochs(args, dataset, model, logger):

	train_batches, validation_batches, test_batches = dataloader_factory.get_batches(args, dataset)

	start_epoch = 0
	stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0}
	for epoch in range(start_epoch, args.num_epochs):
		train_loss = AverageMeter()
		epoch_time = Timer()

		# Run 1 epoch
		for iteration in range(train_batches):

			batch = train_batches[iteration]
			train_loss.update(*model.update(batch))

			if (iteration + 1) % args.eval_interval == 0:
				logger.info('train: Epoch = %d | iter = %d/%d | ' %
						(epoch, iteration, len(train_batches)) +
						'loss = %.2f | elapsed time = %.2f (s)' %
						(train_loss.avg, stats['timer'].time()))
				train_loss.reset()

		logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
						(epoch, epoch_time.time()))

		# Checkpoint model after every epoch
		if args.checkpoint:
			model.checkpoint(args.model_file + ".checkpoint", epoch + 1)


		# Validate Partial Train data
		validate(args, train_batches, model, stats, logger, mode = "train")

		# Validate on Development data
		result = validate(args, validation_batches, model, stats, logger, mode = "dev")

		if result[args.valid_metric] > stats['best_valid']:
			logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
						(args.valid_metric, result[args.valid_metric],
						 stats['epoch'], model.updates))
			model.save(args.model_file)
			stats['best_valid'] = result[args.valid_metric]

		# Recreate training batches using shuffle
		logger.info("Creating train batches for epoch {0}".format(epoch+1))
		train_batches, _, _ = dataloader_factory.get_batches(args, dataset)


def validate(args, batches, model, stats, logger, mode = "dev"):

	eval_time = Timer()
	metrics = MultiTaskAverageMeter(args.metric)

	examples = 0
	for iteration in range(len(batches)):

		batch = batches[iteration]
		examples += len(batch)
		pred, mask  = model.predict(batch)
		target,_ = model.target(batch)

		update_metrics = model.evaluate_metrics(pred, target, mask, mode)
		metrics.update(update_values=update_metrics)

		if mode == 'train' and examples >= 1e4:
			break

	logger.info('%s valid : Epoch = %d | accuracy = %s | ' %
				(mode, stats['epoch'], metrics.print_values()) +
				'examples = %d | ' %
				(examples) +
				'valid time = %.2f (s)' % eval_time.time())





