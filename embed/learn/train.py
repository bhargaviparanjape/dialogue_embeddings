import torch
from torch import optim
from embed.dataloaders import factory as dataloader_factory
from embed.models import factory as model_factory
from embed.models.blocks.embeddings import ELMoEmbedding
from embed.learn import optimizers, metrics
from allennlp.modules.elmo import Elmo, batch_to_ids
import logging

class LearnerState:
	def __init__(self):
		self.train_loss = 0
		self.train_denom = 0
		self.validation_history = []
		self.bad_counter = 0

	def get_loss(self):
		if self.train_denom != 0:
			return self.train_loss/self.train_denom
		else:
			return None

def train(args, dataset, model, logger):
	vars(args)["vocabulary"] = dataset.vocabulary.vocabulary
	train_batches, validation_batches, test_batches = dataloader_factory.get_batches(args, dataset)
	embedding_layer = model_factory.get_embeddings(args, args.embedding, logger)
	# embedding_layer = ELMoEmbedding(args)


	clip_threshold = args.clip_threshold
	eval_interval = args.eval_interval
	patience = args.patience

	optimizer = optimizers.get_optimizer(args, model).optimizer
	learning_state = LearnerState()
	train_metric = metrics.get_metric(args)

	print(len(train_batches))
	for epoch in range(args.num_epochs):
		logger.info("Starting epoch {}".format(epoch + 1))
		for iteration in range(len(train_batches)):
			optimizer.zero_grad()
			if (iteration + 1) % eval_interval == 0:
				logger.info("epoch: {0} iteration: {1} train loss: {2}".format(epoch + 1, iteration + 1, learning_state.get_loss()))
				dev_accuracy = eval(args, validation_batches, model).compute_metric()

				train_accuracy = train_metric.compute_metric()
				learning_state.validation_history.append(dev_accuracy)
				logger.info("epoch: {0} iteration: {1} dev accuracy: {2}".format(epoch + 1, iteration + 1, dev_accuracy))
				logger.info("epoch: {0} iteration: {1} train accuracy: {2}".format(epoch + 1, iteration + 1, train_accuracy))

			batch = train_batches[iteration]
			batch_size, gold_output, mask, *input = model.prepare_for_gpu(batch, embedding_layer)

			loss, *output = model(input, batch_size)

			loss.backward()
			# gradient clipping
			torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
			optimizer.step()

			loss, output = model.prepare_for_cpu(loss, *output)
			if args.use_cuda:
				mask = mask.data.cpu()
				gold_output = gold_output.data.cpu()
			else:
				mask = mask.data
				gold_output = gold_output.data


			learning_state.train_denom += batch_size
			learning_state.train_loss += loss
			train_metric.update_metric(batch_size, output, gold_output, mask)

		logger.info("Saving model after epoch {0}".format(epoch+1))
		torch.save(model, args.model_path + ".epoch{0}.temp".format(epoch+1))

		logger.info("Creating train batches for epoch {0}".format(epoch + 2))

		train_batches,_,_ = dataloader_factory.get_batches(args, dataset)


def eval(args, batches, model, mode="dev"):
	model.train(False)
	dev_metric = metrics.get_metric(args)
	embedding_layer = ELMoEmbedding(args)
	for iteration in range(len(batches)):
		batch = batches[iteration]
		batch_size, gold_output, mask, *input = model.prepare_for_gpu(batch, embedding_layer)

		indices = model.eval(input, batch_size)
		if args.use_cuda:
			indices = indices.data.cpu()
			mask = mask.data.cpu()
			gold_output = gold_output.data.cpu()
		else:
			indices = indices.data
			mask = mask.data
			gold_output = gold_output.data

		# model.prepare_for_cpu(None, output)
		dev_metric.update_metric(batch_size, indices, gold_output, mask)
	model.train(True)
	return dev_metric

def evaluate(args, dataset, model, logger):
	_, validation_batches, test_batches = dataloader_factory.get_batches(args, dataset)
	validation_metric = eval(validation_batches, model, mode="dev")
	test_metric = eval(test_batches, model, mode="test")



