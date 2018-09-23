from datasets.swda import swda
from embed.dataloaders.AbstractDataset import AbstractDataset
from embed.dataloaders.factory import RegisterDataset
from embed.utils.vocabulary import Vocabulary
import random

DAMSL_TAGSET = {'%':0,'+':1,'^2':2,'^g':3,'^h':4,'^q':5,'aa':6,'aap_am':7,'ad':8,'ar':9,'arp_nd':10,'b':11,'b^m':12
	,'ba':13,'bd':14,'bf':15,'bh':16,'bk':17,'br':18,'fa':19,'fc':20,'fo_o_fw_"_by_bc':21,'fp':22,'ft':23,'h':24
	,'na':25,'ng':26,'nn':27,'no':28,'ny':29,'oo_co_cc':30,'qh':31,'qo':32,'qrr':33,'qw':34,'qw^d':35,'qy':36,'qy^d':37
	,'sd':38,'sv':39,'t1':40,'t3':41,'x':42}

@RegisterDataset('swda')
class SwitchBoard(AbstractDataset):
	class Utterance:
		## minimum elements all datasets must have; id, length, tokens
		def __init__(self, id, utterance):
			self.index = utterance.utterance_index
			self.id = id
			self.label = DAMSL_TAGSET[utterance.damsl_act_tag().strip()] # index for DAMSL starts from 1
			self.speaker = utterance.caller
			#TODO: clean text before processing
			self.tokens = utterance.text_words()
			self.length = len(self.tokens)
			self.pos = utterance.regularize_pos_lemmas()

	class Dialogue:
		## minimum elements all datasets must have; id, length, utterances
		def __init__(self, transcript):
			self.id = transcript.conversation_no
			## length of transcript not same as number of utterances
			self.length = transcript.length
			self.conversation_topic = transcript.topic_description
			self.utterances = []
			for id, utterance in enumerate(transcript.utterances):
				self.utterances.append(SwitchBoard.Utterance(id, utterance))

	def __init__(self, args, dataset_path):
		corpus = swda.CorpusReader(dataset_path)
		self.total_length = 0
		self.vocabulary = Vocabulary()

		dataset = []
		for transcript in corpus.iter_transcripts(display_progress=True):
			if args.truncate_dataset and self.total_length > 25:
				break
			dataset.append(SwitchBoard.Dialogue(transcript))
			self.total_length += 1

		shuffled_dataset = random.shuffle(dataset)

		## 1155 transcribed datapoints ; 1115, 19, 21 split
		if args.truncate_dataset:
			self.train_dataset = dataset[:15]
			self.valid_dataset = dataset[15:20]
			self.test_dataset = dataset[20:]
		else:
			##TODO: this split adheres to numbers reporteed by Schriberg et. al., but ideally cross-validation should be done
			self.train_dataset = dataset[:1115]
			self.valid_dataset = dataset[1115:1134]
			self.test_dataset = dataset[1134:]

		## create vocabulary from training data (unks  during test time)
		for data_point in self.train_dataset:
			for utterance in data_point.utterances:
				self.vocabulary.add_and_get_indices(utterance.tokens)

		## create character vocabulary
		self.vocabulary.get_character_vocab()

