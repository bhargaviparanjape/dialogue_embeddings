import tqdm,os,sys,logging
import torch
from random import shuffle
from src.dataloaders import  factory as dataloader_factory
from src.models import factory as model_factory
from src.utils.utility_functions import AverageMeter,Timer, MultiTaskAverageCounter
from random import shuffle
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def train_epochs_multiplex(args, dataset, model):
	pass

def train_epochs_sharedloss(args, dataset, model):
	train_dataloader, validation_dataloader, test_dataloader = dataloader_factory.get_dataloader(args, dataset, model)

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
		validate(args, train_dataloader, model, stats, mode="train")

		# Validate on Development data
		# TODO: Make this more generic and abstract out model file
		result = validate(args, validation_dataloader, model, stats, mode="dev")

		if result > stats['best_valid']:
			logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
						(" ".join(args.metric), result,
						 stats['epoch'], model.updates))
			model.save()
			stats['best_valid'] = result

		# Recreate training batches using shuffle
		logger.info("Creating train batches for epoch {0}".format(epoch + 1))


def train_epochs(args, dataset, model):
	train_dataloader, validation_dataloader, test_dataloader = dataloader_factory.get_dataloader(args, dataset, model)
	writer = SummaryWriter(args.tensorboard_dir)

	train_sample = []
	examples = 0
	for iteration, batch in enumerate(train_dataloader):
		train_sample.append(batch)
		examples += len(batch)
		if examples > 1e4:
			break

	dev_probe_sample = []
	examples = 0
	for iteration, batch in enumerate(train_dataloader):
		dev_probe_sample.append(batch)
		examples += len(batch)
		if examples > 1e4:
			break

	start_epoch = 0
	stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0}
	bad_counter = 0
	for epoch in range(start_epoch, args.num_epochs):
		stats['epoch'] = epoch
		train_loss = AverageMeter()
		epoch_time = Timer()

		# Run 1 epoch
		for iteration, batch in enumerate(train_dataloader):
			loss = model.update(batch)
			train_loss.update(*loss)

			if (iteration + 1) % args.eval_interval == 0:
				logger.info('train: Epoch = %d | iter = %d/%d | ' %
						(epoch, iteration, len(train_dataloader)) +
						'loss = %.2f | elapsed time = %.2f (s)' %
						(train_loss.avg, stats['timer'].time()))
				writer.add_scalar('Loss', train_loss.avg, iteration + epoch*len(train_dataloader))
				train_loss.reset()

			if (iteration + 1) % args.save_interval == 0:
				logger.info('Saving model at epoch {0}, iteration {1}' % (epoch, iteration))
				validate(args, dev_probe_sample, model, stats, mode = "dev")
				model.save()

		logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
						(epoch, epoch_time.time()))

		# Checkpoint model after every epoch
		if args.checkpoint:
			model.checkpoint(args.model_file + ".checkpoint", epoch + 1)


		# Validate Partial Train data
		validate(args, train_sample, model, stats, mode = "train")

		# Validate on Development data
		# TODO: Make this more generic and abstract out model file
		result = validate(args, validation_dataloader, model, stats, mode = "dev")

		writer.add_scalar(model.args.valid_metric, result, epoch)

		if result > stats['best_valid']:
			logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
						(model.args.valid_metric, result,
						 stats['epoch'], model.updates))
			model.save()
			stats['best_valid'] = result
			bad_counter = 0
		else:
			bad_counter += 1
			logger.info("Bad Counter Incremented to %d" % bad_counter)
			if bad_counter > 30:
				logger.info("Early stopping after %d epochs" % epoch)
				logger.info("Best Result : %.4f" % stats['best_valid'])
				bad_counter = 0
				# exit(0)
		# Recreate training batches using shuffle
		logger.info("Creating train batches for epoch {0}".format(epoch+1))


def validate(args, dataloader, model, stats, mode = "dev"):

	eval_time = Timer()
	metrics = MultiTaskAverageCounter(args.models)

	examples = 0
	for iteration, batch in enumerate(dataloader):
		examples += len(batch)
		pred, mask  = model.predict(batch)
		target,_ = model.target(batch)

		update_metrics = {}
		update_metrics[args.models[0]["model"]] = model.evaluate_metrics(pred, target, mask, mode)
		metrics.update(update_values=update_metrics)

		if mode == 'train' and examples >= 1e4:
			break

	logger.info('%s valid : Epoch = %d | metrics = %s [' %
				(mode, stats['epoch'], metrics.print_values()) +
				'] | examples = %d | ' %
				(examples) +
				'valid time = %.2f (s)' % eval_time.time())

	# The valid-metric that will be considered in the validation history and used for patience computation;
	# Valid metric of multiple tasks will be averaged out
	validation_metrics = [(model['model'], model["valid_metric"]) for model in args.models]
	result = metrics.validation_metric(validation_metrics)
	return result


