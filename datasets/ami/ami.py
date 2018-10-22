import glob
import os,sys,pdb,numpy as np,json,time,re,datetime,collections,csv,pdb
from collections import defaultdict
from bs4 import BeautifulSoup
import codecs
import nltk

src_dirname = os.path.dirname(os.path.realpath(__file__))
MANUAL_TRANSCRIPTION_DIRECTORY = os.path.join(src_dirname, "manual/words")
AUTOMATIC_TRANSCRIPTION_DIRECTORY = os.path.join(src_dirname, "automatic/ASR/ASR_AS_CTM_v1.0_feb07")
MANUAL_DIALOGUEACT_DIRECTORY = os.path.join(src_dirname, "manual/dialogueActs")

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
	def __init__(self, ami_filename):
		self.ami_filename = ami_filename
		rows = list(csv.reader(open(self.ami_filename, 'rt'), delimiter="\t"))
		self.header = rows[0]
		rows.pop(0)
		self.conversation_no = os.path.split(ami_filename)[1].split('.', 1)[0]
		self.utterances = [Utterance(x) for x in rows]

class Utterance:
	header = [
		"utterance_id",
		"speaker",
		"start_time",
		"end_time",
		"tokens",
		"dialogue_act"
	]

	def __init__(self, row):
		for i in range(len(Utterance.header)):
			att_name = Utterance.header[i]
			row_value = row[i].strip()
			if att_name == "tokens":
				row_value = self.tokenize(row_value)
			setattr(self, att_name, row_value)

	## TODO: Add scripts to clean up the transcripts according to transcription manual specifications

	def tokenize(self, utterance_text):
		return nltk.word_tokenize(utterance_text)

if __name__ == '__main__':
	words_content = dict()
	dialogue_act_content = dict()
	for filepath in glob.glob(os.path.join(MANUAL_TRANSCRIPTION_DIRECTORY, "*.xml")):
		filename = os.path.split(filepath)[1]
		group_id = filename.split('.')[0]
		person_id = filename.split('.')[1]
		if group_id not in words_content:
			words_content[group_id] = {}
		words_content[group_id][person_id] = open(filepath).read()
	for filepath in glob.glob(os.path.join(MANUAL_DIALOGUEACT_DIRECTORY, "*.xml")):
		filename = os.path.split(filepath)[1]
		group_id = filename.split('.')[0]
		person_id = filename.split('.')[1]
		if group_id not in dialogue_act_content:
			dialogue_act_content[group_id] = {}
		dialogue_act_content[group_id][person_id] = open(filepath).read()
	## Some transcripts have no dialogue acts available

	regex_da = re.compile("ami_da_[0-9]+")
	regex_words = re.compile("words[x]*[0-9]+")

	ami_utterances = {}
	ami_words = {}
	for id in words_content:
		group_utterances = {}
		group_words = {}
		for person in words_content[id]:
			group_utterances[person] = {}
			group_words[person] = {}
			soup = BeautifulSoup(words_content[id][person])
			word_tags = soup.find_all('w')
			vocal_tags = [child for child in list(soup.find('nite:root').children) if child.name != "w" and child != '\n']
			full_tags = word_tags + vocal_tags
			current_utterance = []
			tokens = []
			word_range = []
			for i, word in enumerate(word_tags):
				word_id = int(regex_words.findall(word['nite:id'])[0].replace("words", "").replace("x", ""))
				start_time = float(word.get('starttime', 500.0))
				end_time = float(word.get('endtime', 500.0))
				text = word.text
				group_words[person][word_id] = [person, start_time, end_time, text]
				if len(tokens) == 0:
					current_start = start_time
					word_range.append(word_id)
				tokens.append(text)
				if 'punc' in word.attrs and text in ['.', '?', '!']:
					current_end = end_time
					current_text = tokens
					word_range.append(word_id)
					group_utterances[person][tuple(word_range)] = [person, current_start, current_end, " ".join(current_text)]
					tokens = []
					word_range = []
			for i, vocal in enumerate(vocal_tags):
				if len(regex_words.findall(vocal['nite:id'])) == 0:
					pdb.set_trace()
				vocal_id = int(regex_words.findall(vocal['nite:id'])[0].replace("words", "").replace("x", ""))
				if "starttime" in vocal:
					start_time = float(vocal['starttime'])
				else:
					start_time = 500.0
				if "endtime" not in vocal:
					entime = start_time
				else:
					end_time = float(vocal['endtime'])
				if vocal.name == "vocalsound":
					text = "(" + vocal['type'] + ")"
				else:
					text = ""
				word_range = [vocal_id]
				group_utterances[person][tuple(word_range)] = [person, start_time, end_time, text]
				group_words[person][vocal_id] = [person, start_time, end_time, text]
			## order group_utterances[person] ()
			# for the vocal processors that occur within an utterance add that to the utterance itself
			group_utterances[person] = dict(collections.OrderedDict(sorted(group_utterances[person].items())))
		ami_words[id] = group_words
		ami_utterances[id] = group_utterances



	dialogue_act_information = {}
	for id in dialogue_act_content:
		dialogue_act_information[id] = {}
		for person in dialogue_act_content[id]:
			dialogue_act_information[id][person] = {}
			soup = BeautifulSoup(dialogue_act_content[id][person])
			act_tags = soup.find_all('dact')
			for act in act_tags:
				for child in act.children:
					# href="da-types.xml#id(ami_da_3)"
					if child.name == "nite:pointer":
						dialogue_act = int(regex_da.findall(child['href'])[0].split('_')[-1])
					# ES2002a.A.words.xml#id(ES2002a.A.words299)..id(ES2002a.A.words301)
					if child.name == "nite:child":
						word_range = [int(s.replace("words", "")) for s in regex_words.findall(child['href'])]
				dialogue_act_information[id][person][tuple(word_range)] = dialogue_act

	## match the word order tuples and assign a dialogue act to each utterance
	for id in ami_utterances:
		# if id not in dialogue act information: denote with special marke -1
		if id not in dialogue_act_information:
			# list of ami_utterances[id] by id
			full_utterance_list = []
			for person in ami_utterances[id]:
				full_utterance_list += ami_utterances[id][person].values()
			full_utterance_list = [l + [-1] for l in full_utterance_list]
			ordered_utterance_list = sorted(full_utterance_list, key=lambda x : x[1])
				## add dummy dialogue acts
		else:
			full_utterance_list = []
			for person in ami_utterances[id]:
				utterance_words = ami_words[id][person]
				dialogue_acts = dialogue_act_information[id][person]
				final_utterance_list = []
				for key in dialogue_acts:
					if len(key) == 1:
						final_utterance_list.append(utterance_words[key[0]] + [dialogue_acts[key]])
					else:
						utterance = []
						person = utterance_words[key[0]][0]
						start_time = utterance_words[key[0]][1]
						end_time = utterance_words[key[1]][2]
						for index in range(key[0], key[1]+1):
							if index in utterance_words:
								utterance.append(utterance_words[index][3])
							else:
								utterance.append("")
						utterance = " ".join(utterance)
						final_utterance_list.append([person, start_time, end_time, utterance] + [dialogue_acts[key]])
				full_utterance_list += final_utterance_list
			ordered_utterance_list = sorted(full_utterance_list, key=lambda x : x[1])

		with codecs.open(os.path.join(MANUAL_TRANSCRIPTION_DIRECTORY, id + ".csv"), "w+", encoding="utf-8") as fout:
			fout.write("utterance_id\tspeaker\tstart_time\tend_time\tutterance_text\tdialogue_act")
			for u_id, utterance in enumerate(ordered_utterance_list):
				# utterance[-2] = " ".join(utterance[-2])
				utterance = [u_id + 1] + utterance
				fout.write("%d\t%s\t%f\t%f\t%s\t%d\n" % tuple(utterance))




