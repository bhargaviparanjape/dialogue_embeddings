import glob
import os,sys,pdb,numpy as np,json,time,re,datetime,csv
from collections import defaultdict
from bs4 import BeautifulSoup
import nltk
import codecs

src_dirname = os.path.dirname(os.path.realpath(__file__))
CORNELLMOVIE_TRANSCRIPTION_DIRECTORY = os.path.join(src_dirname, "data")

class CorpusReader:
	def __init__(self, src_dirname):
		self.src_dirname = src_dirname

	def iter_transcripts(self, display_progress=True):
		i = 1
		data = open(os.path.join(self.src_dirname,
		                CORNELLMOVIE_TRANSCRIPTION_DIRECTORY, "movie_lines.txt")).readlines()
		for datapoint in
			if display_progress:
				sys.stderr.write("\r")
				sys.stderr.write("transcript %s" % i)
				sys.stderr.flush()
				i += 1
			yield Transcript(filename)
			# Closing blank line for the progress bar:
		if display_progress: sys.stderr.write("\n")


if __name__ == "__main__":
	'''
	- fields:
		- lineID
		- characterID (who uttered this phrase)
		- movieID
		- character name
		- text of the utterance
	'''
	data =  open(os.path.join(src_dirname,
		                CORNELLMOVIE_TRANSCRIPTION_DIRECTORY, "movie_lines.txt")).readlines()

