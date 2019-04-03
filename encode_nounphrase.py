import h5py
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nounphrasemapping', dest='nounphrasemapping', default='', type=str)
parser.add_argument('--vocabmapping', dest='vocabmapping', default='', type=str)
parser.add_argument('--dest', dest='dest', default='', type=str)

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(1)

args = parser.parse_args()

nounphrase_mappings = json.load(open(args.nounphrasemapping))

vocab_mappings = json.load(open(args.vocabmapping))

input_wtoi = vocab_mappings["input_wtoi"]
n = len(nounphrase_mappings["noun_itow"])

encode = np.zeros((n, 18))

for k, v in nounphrase_mappings["noun_itow"].iteritems():
	words = v.split(" ")
	for i, word in enumerate(words):
		encode[int(k) - 1][i] = input_wtoi.get(word, input_wtoi["UNK"])

f = h5py.File(args.dest, "w")
f.create_dataset("encode", data=encode, dtype="uint32")
f.close()
