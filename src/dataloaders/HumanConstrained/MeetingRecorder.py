from datasets.mrda import mrda
from src.dataloaders.AbstractDataset import AbstractDataset
from src.dataloaders.HumanSpontaneous.SwitchBoard import DAMSL_TAGSET
from src.dataloaders.factory import RegisterDataset
from src.utils.vocabulary import Vocabulary
import re,os
import numpy as np
from collections import Counter

MRDA_DAMSL_MAP = { '%' : '%', '%-' : '%--', 'x' : 'x', 't1' : 't1', 't3' : 't3', 't' : 't', 'c' : '##'
				   , 'sd' : 's', 'sv' : 's', 'oo' : '##', 'qy' : 'qy', 'qw' : 'qw' , 'qo' : 'qo', 'qr' : 'qr'
				   , 'qrr' : 'qrr', 'qh' : 'qh', 'd' : 'd', 'g' : 'g', 'ad' : 'co', 'co' : 'cs', 'cc' : 'cc'
				   , 'fp' : '##', 'fc' : '##', 'fx' : '##', 'fe' : 'fe', 'fo' : '##', 'ft' : 'ft', 'fw' : 'fw'
				   , 'fa' : 'fa', 'aa' : 'aa', 'aap' : 'aap', 'am' : 'am', 'arp' : 'arp', 'ar' : 'ar', 'h' : 'h'
				   , 'br' : 'br', 'b' : 'b' , 'bh' : 'bh', 'bk' : 'bk', 'm' : 'm', '2' : '2', 'bf' : 'bs', 'ba' : 'ba'
				   , 'by' : 'by', 'bd' : 'bd', 'bc' : 'bc', 'ny' : 'aa', 'nn' : 'ar', 'na' : 'na', 'ng' : 'ng'
				   , 'no' : 'no', 'e' : 'e', 'nd' : 'nd', 'q' : '##', 'h' : '##', '+' : '##'}

SIMPLE_TAGS = {
	"S": 0, "B" : 1, "Q" :2, "F": 3, "D" : 4, "Z" : 5
}

MRDA_GENERAL_TAGS = {
	"x" : 0, "s" : 1, "qy": 2, "qw": 3, "qr":4, "qrr":5, "qo":6, "qh":7, "b":8, "fg":9, "fh":10, "h":11
}

MRDA_SPECIAL_TAGS = {
	"aa" : 0, "aap" : 1, "am" : 2, "ar" : 3, "arp" : 4, "ba" : 5, "bc" : 6, "bd" : 7, "bh" : 8, "bk" : 9, "br" : 10,
	"bs" : 11, "bsc" : 12, "bu" : 13, "by" : 14, "cc" : 15, "co" : 16, "cs" : 17, "d" : 18, "df" : 19, "e" : 20,
	"f" : 21, "fa" : 22, "fe" : 23, "ft" : 24, "fw" : 25, "g" : 26, "j" : 27, "m" : 28, "na" : 29, "nd" : 30,
	"ng" : 31, "no" : 32, "r" : 33,	"rt" : 34, "t" : 35, "tc" : 36, "t1" : 37, "t3" : 38, "2" : 39
}

# Standard dataset splits
TRAIN_SPLIT = ['Bdb001', 'Bed002', 'Bed004', 'Bed005', 'Bed008', 'Bed009', 'Bed011', 'Bed013', 'Bed014', 'Bed015', 'Bed017', 'Bmr002', 'Bmr003', 'Bmr006', 'Bmr007', 'Bmr008', 'Bmr009', 'Bmr011', 'Bmr012', 'Bmr015', 'Bmr016', 'Bmr020', 'Bmr021', 'Bmr023', 'Bmr025', 'Bmr026', 'Bmr027', 'Bmr029', 'Bmr031', 'Bns001', 'Bns002', 'Bns003', 'Bro003', 'Bro005', 'Bro007', 'Bro010', 'Bro012', 'Bro013', 'Bro015', 'Bro016', 'Bro017', 'Bro019', 'Bro022', 'Bro023', 'Bro025', 'Bro026', 'Bro028', 'Bsr001', 'Btr001', 'Btr002', 'Buw001']
DEV_SPLIT = ['Bed003', 'Bed010', 'Bmr005', 'Bmr014', 'Bmr019', 'Bmr024', 'Bmr030', 'Bro004', 'Bro011', 'Bro018', 'Bro024']
TEST_SPLIT =  ['Bed006', 'Bed012', 'Bed016', 'Bmr001', 'Bmr010', 'Bmr022', 'Bmr028', 'Bro008', 'Bro014', 'Bro021', 'Bro027']


@RegisterDataset('mrda')
class MeetingRecoder(AbstractDataset):
	class Utterance:
		def __init__(self, utterance, tag_map):
			self.id = utterance.utterance_id
			##mapping between DAMSL and tagset used in SWDA
			da_tag = utterance.da_tag.strip()
			if da_tag != '':
				self.label = SIMPLE_TAGS[tag_map[utterance.da_tag.strip()]]
			else:
				self.label = SIMPLE_TAGS['Z']
			self.speaker = utterance.speaker
			self.tokens = utterance.original_text
			self.text = " ".join(self.tokens)
			self.length = len(self.tokens)

	class Dialogue:
		def __init__(self, transcript, tag_map):
			self.id = "mrda_" + str(transcript.conversation_id)
			self.conversation_length = len(transcript.utterances)
			self.utterances = []
			for utterance in transcript.utterances:
				## only consider data subset that can be tagged with mrda damsl tags
				# if utterance.da_tag in MRDA_DAMSL_MAP and MRDA_DAMSL_MAP[utterance.da_tag] != "##" and MRDA_DAMSL_MAP[utterance.da_tag] in DAMSL_TAGSET:
				self.utterances.append(MeetingRecoder.Utterance(utterance, tag_map))

	@staticmethod
	def processs_mrda_tag(tag_string, type="simple"):
		# tags are separated by |, :
		sub_utterance_tags = re.split('\||:', tag_string)
		# choose first subtag and split by ^, pick first part as primary DA tag
		if len(sub_utterance_tags) > 1:
			# primarily DA information maybe in the second clause
			subtags = sub_utterance_tags[1].split('^')
		else:
			subtags = sub_utterance_tags[0].split('^')
		primary_tag = subtags[0]
		if len(subtags) > 1:
			first_secondary_tag = subtags[1]
		# disruption
		if "%" in tag_string:
			simple_tag = "D"
		# statements and whatever h is
		elif primary_tag == "s" or primary_tag == "h":
			simple_tag = "S"
		# floor grabbing
		elif primary_tag == "fg" or primary_tag == "fh":
			simple_tag = "FG"
		# backchanneling
		elif primary_tag == "b":
			simple_tag = "B"
		# question
		elif "q" in primary_tag:
			simple_tag = "Q"
		else:
			# catching "z" (not marked); ""(missing), "x": non-speech : into a separate category
			simple_tag ="X"

		if type == "general":
			return MRDA_GENERAL_TAGS[primary_tag]
		elif type == "special":
			return MRDA_SPECIAL_TAGS[first_secondary_tag]
		elif type == "simple":
			return SIMPLE_TAGS[simple_tag]

	def __init__(self, args, dataset_path):
		self.name = type(self).__name__
		corpus = mrda.CorpusReader(dataset_path)
		# train, test splits standard
		self.total_length = 0
		self.vocabulary = Vocabulary()
		tag_classmap = open(os.path.join(dataset_path, "map_01b_expanded")).readlines()
		self.tag_classmap = {}
		for line in tag_classmap:
			components  = line.split("\t")
			self.tag_classmap[components[0].strip()] = components[1].strip()
		self.label_set_size = len(SIMPLE_TAGS)

		dataset = []
		for transcript in corpus.iter_transcripts(display_progress=True):
			self.total_length += 1
			if args.truncate_dataset and self.total_length > 20:
				break
			dataset.append(MeetingRecoder.Dialogue(transcript, self.tag_classmap))

		#TODO: Exact test-dev split for mrda //actually do cross validation
		if args.truncate_dataset:
			self.train_dataset = dataset[:10]
			self.valid_dataset = dataset[10:15]
			self.test_dataset = dataset[15:20]
		else:
			# Split used by the state of the art
			self.train_dataset = []
			self.valid_dataset = []
			self.test_dataset = []
			for dialogue in dataset:
				idx = dialogue.id.split("_")[1]
				if idx in TRAIN_SPLIT:
					self.train_dataset.append(dialogue)
				elif idx in DEV_SPLIT:
					self.valid_dataset.append(dialogue)
				elif idx in TEST_SPLIT:
					self.test_dataset.append(dialogue)
				else:
					pass

		if args.simulate_low_resource != None:
			num_reduced_samples = int(args.simulate_low_resource*len(self.train_dataset))
			self.train_dataset = np.random.choice(self.train_dataset, num_reduced_samples, replace=False).tolist()


		## create vocabulary from training data (UNKS  during test time)
		for data_point in self.train_dataset:
			for utterance in data_point.utterances:
				self.vocabulary.add_and_get_indices(utterance.tokens)

		## create character vocabulary
		self.vocabulary.get_character_vocab()
		self.utterance_length = self.get_total_utterances()
