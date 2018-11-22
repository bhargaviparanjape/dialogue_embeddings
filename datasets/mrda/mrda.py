#!/usr/bin/env python

import csv
import datetime
import os
import re
import sys
import glob
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer

class CorpusReader:

	def __init__(self, src_dirname):
		self.src_dirname = src_dirname
		self.metadata = None
	def iter_transcripts(self, display_progress=True):
		i = 1
		for filename in glob.glob(os.path.join(self.src_dirname, "*.trans")):
			if display_progress:
				sys.stderr.write("\r")
				sys.stderr.write("transcript %s" % i)
				sys.stderr.flush();
				i += 1
			yield Transcript(filename, self.metadata)
		if display_progress: sys.stderr.write("\n")

	def iter_utterances(self, display_progress=True):
		"""
		Iterate through the utterances.

		Parameters
		----------
		display_progress : bool (default: True)
			Display an overwriting progress bar if True.
		"""
		i = 1
		for trans in self.iter_transcripts(display_progress=False):
			for utt in trans.utterances:
				# Optional progress bar.
				if display_progress:
					sys.stderr.write("\r")
					sys.stderr.write("utterance %s" % i)
					sys.stderr.flush();
					i += 1
				# Yield the Utterance instance:
				yield utt
		# Closing blank line for the progress bar:
		if display_progress: sys.stderr.write("\n")


class Transcript:
	def __init__(self, mrda_filename, metadata):
		self.dadb_filename = mrda_filename.rsplit(".", 1)[0] + ".dadb"
		self.trans_filename = mrda_filename

		transcript_rows = list(csv.reader(open(self.trans_filename, 'rt')))
		databse_rows = list(csv.reader(open(self.dadb_filename, 'rt')))
		assert (len(transcript_rows) == len(databse_rows))
		self.conversation_id = mrda_filename.split('.', 1)[0].split('/')[-1]
		## TODO: some utterances maybe ignored by explicit instruction of the annotators
		self.utterances = [Utterance(x) for x in zip(transcript_rows, databse_rows)]


class Utterance:

	header1 = [
		'start_time',
		'end_time',
		'utterance_id',
		'error_code',
		'internal_word_times',
		'da_tag',
		'channel_info',
		'speaker',
		'original_da_tag',
		'adjacency_pair',
		'original_adjacency_pair',
		'hot_spot',
		'hot_spot_comment',
		'da_ap_comment'
	]
	header2 = [
		'utterance_id',
		'original_text',
		'text'
	]
	def __init__(self, row_tuple):
		db = row_tuple[1]
		trans = row_tuple[0]
		for i in range(len(Utterance.header1)):
			att_name = Utterance.header1[i]
			row_val = db[i]
			if att_name == "start_time" or att_name == "end_time":
				row_val = float(row_val)
			setattr(self, att_name, row_val)
		for i in range(len(Utterance.header2)):
			att_name = Utterance.header2[i]
			row_val = trans[i]
			if att_name.endswith("text"):
				row_val = [x.strip() for x in row_val.split()]
			setattr(self, att_name, row_val)






