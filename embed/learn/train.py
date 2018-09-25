import torch
from torch import optim
from embed.dataloaders import factory as dataloader_factory
from embed.models import factory as model_factory
from embed.models.factory import variable, FloatTensor, ByteTensor, LongTensor
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

def generate_embeddings(args, dataset, model, logger):
	## run through all 3 batches and dump by conversation_id and utterance_id
	model.train(False)
	vars(args)["vocabulary"] = dataset.vocabulary.vocabulary
	train_batches, validation_batches, test_batches = dataloader_factory.get_batches(args, dataset)
	embedding_layer = model_factory.get_embeddings(args, args.embedding, logger)

	combined_batches = [train_batches, validation_batches, test_batches]
	for batches in combined_batches:
		for iteration in range(len(batches)):
			batch = batches[iteration]
			batch_size, gold_output, mask, *input = model.prepare_for_gpu(batch, embedding_layer)
			embeddings = model.dump_embeddings(input, batch_size)
	model.train(True)

def train(args, dataset, model, logger):

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
		train_metric.reset()
		for iteration in range(len(train_batches)):
			optimizer.zero_grad()
			if (iteration + 1) % eval_interval == 0:
				logger.info("epoch: {0} iteration: {1} train loss: {2}".format(epoch + 1, iteration + 1, learning_state.get_loss()))
				dev_accuracy = eval(args, validation_batches, model, embedding_layer).compute_metric()

				train_accuracy = train_metric.compute_metric()
				learning_state.validation_history.append(dev_accuracy)
				logger.info("epoch: {0} iteration: {1} dev accuracy: {2}".format(epoch + 1, iteration + 1, dev_accuracy))
				logger.info("epoch: {0} iteration: {1} train accuracy: {2}".format(epoch + 1, iteration + 1, train_accuracy))

				if dev_accuracy >= max(learning_state.validation_history):
					print("Saving best model seen so far epoch {0}, itr number {1}".format(epoch + 1, iteration + 1))
					torch.save(model, args.model_path)
					print("Best on Validation: Accuracy:{0}".format(dev_accuracy))
					learning_state.bad_counter = 0
				else:
					learning_state.bad_counter += 1
				if learning_state.bad_counter > patience:
					print("Early Stopping")
					# print("Testing started")
					# model = torch.load(args.model_path)
					# evaluate(args, model, test_batches, vocabulary)
					exit(0)

			batch = train_batches[iteration]
			batch_size, gold_output, mask, *input = model.prepare_for_gpu(batch, embedding_layer)

			loss, *output = model(input, batch_size)

			loss.backward()
			# gradient clipping
			torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
			optimizer.step()

			loss_value, *output = model.prepare_for_cpu(loss, *output)
			if args.use_cuda:
				mask = mask.data.cpu()
				_, *gold_output = model.prepare_for_cpu(loss, gold_output)
			else:
				mask = mask.data
				_, *gold_output = model.prepare_for_cpu(loss, gold_output)


			learning_state.train_denom += batch_size
			learning_state.train_loss += loss_value
			train_metric.update_metric(batch_size, output, gold_output, mask)

		logger.info("Saving model after epoch {0}".format(epoch+1))
		torch.save(model, args.model_path + ".temp")

		logger.info("Creating train batches for epoch {0}".format(epoch + 2))

		train_batches,_,_ = dataloader_factory.get_batches(args, dataset)


def eval(args, batches, model, embedding_layer, mode="dev"):
	model.train(False)
	dev_metric = metrics.get_metric(args)

	dummy_loss = FloatTensor([0])
	for iteration in range(len(batches)):
		batch = batches[iteration]
		batch_size, gold_output, mask, *input = model.prepare_for_gpu(batch, embedding_layer)

		indices = model.eval(input, batch_size)
		_, *indices = model.prepare_for_cpu(dummy_loss, indices)
		_, *gold_output = model.prepare_for_cpu(dummy_loss, gold_output)
		if args.use_cuda:
			mask = mask.data.cpu()
		else:
			mask = mask.data

		# model.prepare_for_cpu(None, output)
		dev_metric.update_metric(batch_size, indices, gold_output, mask)
	model.train(True)
	return dev_metric

def store_embeddings(args, dataset, model, logger):
	## batch and run evaluate and return the dialogue embeddings for that conversation id
	raise NotImplementedError

def evaluate(args, dataset, model, logger):
	_, validation_batches, test_batches = dataloader_factory.get_batches(args, dataset)
	embedding_layer = model_factory.get_model(args, args.embedding, logger)
	validation_metric = eval(validation_batches, model, embedding_layer, mode="dev")
	test_metric = eval(test_batches, model, embedding_layer, mode="test")



