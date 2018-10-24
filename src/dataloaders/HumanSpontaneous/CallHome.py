from datasets.call import call
from src.dataloaders.AbstractDataset import AbstractDataset
from src.dataloaders.factory import RegisterDataset
from src.utils.vocabulary import Vocabulary

@RegisterDataset('call_home_eng')
class CallHomeEnglish(AbstractDataset):
	class Utterance:
		def __init__(self, id, utterance):
			self.name = "call_home_eng"
			self.id = utterance.utterance_id
			self.speaker = utterance.speaker
			self.tokens = utterance.tokens
			self.length = len(self.tokens)
			# assign dummy label
			self.label = 0
			self.start_time = utterance.start_time
			self.end_time = utterance.end_time

	class Dialogue:
		def __init__(self, transcript):
			self.id = transcript.conversation_no
			self.utterances = []
			for id, utterance in enumerate(transcript.utterances):
				self.utterances.append(CallHomeEnglish.Utterance(id, utterance))
			self.length = len(self.utterances)


	def __init__(self, args, dataset_path):
		self.name = type(self).__name__
		corpus = call.CorpusReader(dataset_path)
		self.total_length = 0
		self.label_set_size = 0
		self.vocabulary = Vocabulary()

		dataset = []
		for transcript in corpus.iter_transcripts(display_progress=True):
			if args.truncate_dataset and self.total_length > 25:
				break
			dataset.append(CallHomeEnglish.Dialogue(transcript))
			self.total_length += 1

		if args.truncate_dataset:
			self.train_dataset = dataset[:15]
			self.valid_dataset = dataset[15:20]
			self.test_dataset = dataset[20:]
		else:
			## depending on what task (in args) you can choose to return only a subset that is annotated for DA
			self.train_dataset = dataset[:140]
			self.valid_dataset = dataset[141:155]
			self.test_dataset = dataset[156:]

		for data_point in self.train_dataset:
			for utterance in data_point.utterances:
				self.vocabulary.add_and_get_indices(utterance.tokens)

		if args.limit_vocabulary:
			self.vocabulary.truncate()

		## create character vocabulary
		self.vocabulary.get_character_vocab()