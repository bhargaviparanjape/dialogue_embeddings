import glob
import os,sys,pdb,numpy as np,json,time,re,datetime,csv
from collections import defaultdict
from bs4 import BeautifulSoup
import nltk
import codecs

src_dirname = os.path.dirname(os.path.realpath(__file__))
CALLHOME_TRANSCRIPTION_DIRECTORY = os.path.join(src_dirname, "callhome_eng/eng")

class CorpusReader:
	def __init__(self, src_dirname):
		self.src_dirname = src_dirname

	def iter_transcripts(self, display_progress=True):
		i = 1
		for filename in glob.glob(os.path.join(self.src_dirname, "*.csv")):
			if display_progress:
				sys.stderr.write("\r")
				sys.stderr.write("transcript %s" % i)
				sys.stderr.flush()
				i += 1
			yield Transcript(filename)
			# Closing blank line for the progress bar:
		if display_progress: sys.stderr.write("\n")

	def iter_utterances(self, display_progress=True):
		i = 1
		for trans in self.iter_transcripts(display_progress=False):
			for utt in trans.utterances:
				# Optional progress bar.
				if display_progress:
					sys.stderr.write("\r")
					sys.stderr.write("utterance %s" % i)
					sys.stderr.flush()
					i += 1
				# Yield the Utterance instance:
				yield utt
		# Closing blank line for the progress bar:
		if display_progress: sys.stderr.write("\n")

class Transcript:
	def __init__(self, callhome_filename):
		self.callhome_filename = callhome_filename
		rows = list(csv.reader(open(self.callhome_filename, 'rt'), delimiter='\t'))
		self.header = rows[0]
		rows.pop(0)
		self.conversation_no = os.path.split(callhome_filename)[1].split('.', 1)[0]
		self.utterances = [Utterance(x) for x in rows]

class Utterance:
	header = [
		"utterance_id",
		"speaker",
		"start_time",
		"end_time",
		"tokens",
	]

	def __init__(self, row):
		for i in range(len(Utterance.header)):
			att_name = Utterance.header[i]
			row_value = row[i].strip()
			if att_name == "tokens":
				row_value = self.tokenize(row_value)
			setattr(self, att_name, row_value)

	## TODO: Add scripts to clean up the transcripts according to transcription manual specifications

	def tokenize(self, utterance):
		return nltk.word_tokenize(utterance)

if __name__ == '__main__':
	for filepath in glob.glob(os.path.join(CALLHOME_TRANSCRIPTION_DIRECTORY, "*.cha")):
		filename = os.path.split(filepath)[1]
		group_id = filename.split('.')[0]
		lines = open(filepath).readlines()
		group_utterances = []
		current_tokens = ""
		utterance_id = 1
		for line in lines:
			if line.startswith("*"):
				line = line.replace(u"\u0015", '\t').strip()
				content = line.split('\t')
				speaker = content[0][1]
				# tokens = nltk.word_tokenize(content[1])
				current_tokens += " " + content[1]
				if len(content) > 2:
					start_time = int(content[2].split("_")[0])
					end_time = int(content[2].split("_")[1])
					group_utterances.append([utterance_id, speaker, start_time, end_time, current_tokens.strip()])
					current_tokens = ""
					utterance_id += 1
		sorted_group_utterances = sorted(group_utterances, key= lambda x: x[2])
		with codecs.open(filepath.split('.' , 1)[0] + ".csv", "w+", encoding="utf-8") as fout:
			fout.write("utterance_id\tspeaker\tstart_time\tend_time\tutterance_text")
			for utterance in sorted_group_utterances:
				utterance[-1] = " ".join(utterance[-1])
				fout.write("%d\t%s\t%d\t%d\t%s\n" % tuple(utterance))
