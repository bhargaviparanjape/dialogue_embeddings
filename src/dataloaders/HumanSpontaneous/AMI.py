from datasets.ami import ami
from src.dataloaders.AbstractDataset import AbstractDataset
from src.dataloaders.factory import RegisterDataset
from src.utils.vocabulary import Vocabulary

AMI_DIALOGUE_TAGSET = []

TRAIN_SPLIT = [
"ES2002", "ES2005", "ES2006", "ES2007", "ES2008", "ES2009", "ES2010", "ES2012", "ES2013", "ES2015", "ES2016",
"IS1000", "IS1001", "IS1002", "IS1003", "IS1004", "IS1005", "IS1006", "IS1007", "TS3005", "TS3008", "TS3009",
"TS3010", "TS3011", "TS3012", "EN2001", "EN2003", "EN2004", "EN2005", "EN2006", "EN2009", "IN1001", "IN1002",
"IN1005", "IN1007", "IN1008", "IN1009", "IN1012", "IN1013", "IN1014", "IN1016"
]

DEV_SPLIT = [
	"ES2003", "ES2011", "IS1008", "TS3004", "TS3006", "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011"
]

TEST_SPLIT = [
	"ES2004", "ES2014", "IS1009", "TS3003", "TS3007", "EN2002"
]

@RegisterDataset('ami')
class AmericanMeetingCorpus(AbstractDataset):
	class Utterance:
		def __init__(self, id, utterance):
			self.name = "ami"
			self.id = utterance.utterance_id
			self.label = utterance.dialogue_act
			self.speaker = utterance.speaker
			self.tokens = utterance.tokens
			self.length = len(self.tokens)
			self.start_time = utterance.start_time
			self.end_time = utterance.end_time

	class Dialogue:
		def __init__(self, transcript):
			self.id = "ami_" + str(transcript.conversation_no)
			self.utterances = []
			for id, utterance in enumerate(transcript.utterances):
				self.utterances.append(AmericanMeetingCorpus.Utterance(id, utterance))
			self.length = len(self.utterances)


	def __init__(self, args, dataset_path):
		self.name = type(self).__name__
		corpus = ami.CorpusReader(dataset_path)
		self.total_length = 0
		self.vocabulary = Vocabulary()
		self.label_set_size = len(AMI_DIALOGUE_TAGSET)

		dataset = []
		for transcript in corpus.iter_transcripts(display_progress=True):
			self.total_length += 1
			if args.truncate_dataset and self.total_length > 25:
				break
			dataset.append(AmericanMeetingCorpus.Dialogue(transcript))


		if args.truncate_dataset:
			self.train_dataset = dataset[:15]
			self.valid_dataset = dataset[15:20]
			self.test_dataset = dataset[20:]
		else:
			## depending on what task (in args) you can choose to return only a subset that is annotated for DA
			self.train_dataset = []
			self.valid_dataset = []
			self.test_dataset = []
			for dialogue in dataset:
				if dialogue.id[:-1] in TRAIN_SPLIT:
					self.train_dataset.append(dialogue)
				elif dialogue.id[:-1] in DEV_SPLIT:
					self.valid_dataset.append(dialogue)
				elif dialogue.id[:-1] in TEST_SPLIT:
					self.test_dataset.append(dialogue)

		for data_point in self.train_dataset:
			for utterance in data_point.utterances:
				self.vocabulary.add_and_get_indices(utterance.tokens)

		## create character vocabulary
		self.vocabulary.get_character_vocab()
		self.utterance_length = self.get_total_utterances()
