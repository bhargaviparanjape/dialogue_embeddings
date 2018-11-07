from datasets.mrda import mrda
from src.dataloaders.AbstractDataset import AbstractDataset
from src.dataloaders.HumanSpontaneous.SwitchBoard import DAMSL_TAGSET
from src.dataloaders.factory import RegisterDataset
from src.utils.vocabulary import Vocabulary

MRDA_DAMSL_MAP = { '%' : '%', '%-' : '%--', 'x' : 'x', 't1' : 't1', 't3' : 't3', 't' : 't', 'c' : '##'
				   , 'sd' : 's', 'sv' : 's', 'oo' : '##', 'qy' : 'qy', 'qw' : 'qw' , 'qo' : 'qo', 'qr' : 'qr'
				   , 'qrr' : 'qrr', 'qh' : 'qh', 'd' : 'd', 'g' : 'g', 'ad' : 'co', 'co' : 'cs', 'cc' : 'cc'
				   , 'fp' : '##', 'fc' : '##', 'fx' : '##', 'fe' : 'fe', 'fo' : '##', 'ft' : 'ft', 'fw' : 'fw'
				   , 'fa' : 'fa', 'aa' : 'aa', 'aap' : 'aap', 'am' : 'am', 'arp' : 'arp', 'ar' : 'ar', 'h' : 'h'
				   , 'br' : 'br', 'b' : 'b' , 'bh' : 'bh', 'bk' : 'bk', 'm' : 'm', '2' : '2', 'bf' : 'bs', 'ba' : 'ba'
				   , 'by' : 'by', 'bd' : 'bd', 'bc' : 'bc', 'ny' : 'aa', 'nn' : 'ar', 'na' : 'na', 'ng' : 'ng'
				   , 'no' : 'no', 'e' : 'e', 'nd' : 'nd', 'q' : '##', 'h' : '##', '+' : '##'}

@RegisterDataset('mrda')
class MeetingRecoder(AbstractDataset):
	class Utterance:
		def __init__(self, utterance):
			self.id = utterance.utterance_id
			##mapping between DAMSL and tagset used in SWDA
			self.label = DAMSL_TAGSET[utterance.da_tag.strip()] - 1 # index for DAMSL starts from 1
			self.speaker = utterance.speaker
			self.tokens = utterance.original_text
			self.length = len(self.tokens)

	class Dialogue:
		def __init__(self, transcript):
			self.id = "mrda_" + str(transcript.conversation_id)
			self.conversation_length = len(transcript.utterances)
			self.utterances = []
			for utterance in transcript.utterances:
				## only consider data subset that can be tagged with mrda damsl tags
				if utterance.da_tag in MRDA_DAMSL_MAP and MRDA_DAMSL_MAP[utterance.da_tag] != "##" and MRDA_DAMSL_MAP[utterance.da_tag] in DAMSL_TAGSET:
					self.utterances.append(MeetingRecoder.Utterance(utterance))

	#

	def __init__(self, args, dataset_path):
		self.name = type(self).__name__
		corpus = mrda.CorpusReader(dataset_path)
		# train, test splits standard
		self.total_length = 0
		self.vocabulary = Vocabulary()
		self.label_set_size = len(DAMSL_TAGSET)

		dataset = []
		for transcript in corpus.iter_transcripts(display_progress=True):
			self.total_length += 1
			if args.truncate_dataset and self.total_length > 20:
				break
			dataset.append(MeetingRecoder.Dialogue(transcript))


		#TODO: Exact test-dev split for mrda //actually do cross validation
		if args.truncate_dataset:
				self.train_dataset = dataset[:10]
				self.valid_dataset = dataset[10:15]
				self.test_dataset = dataset[15:20]
		else:
				self.train_dataset = dataset[:45]
				self.valid_dataset = dataset[45:60]
				self.test_dataset = dataset[60:]

		## create vocabulary from training data (UNKS  during test time)
		for data_point in self.train_dataset:
			for utterance in data_point.utterances:
				self.vocabulary.add_and_get_indices(utterance.tokens)

		## create character vocabulary
		self.vocabulary.get_character_vocab()
