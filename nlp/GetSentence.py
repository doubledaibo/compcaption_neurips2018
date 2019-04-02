'''
Extract sentences from dataset json
'''
import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--fin', dest='fin', default='', type=str)
parser.add_argument('--fout', dest='fout', default='', type=str)

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(1)

args = parser.parse_args()

dataset = json.load(open(args.fin))
dataset = dataset["images"]
sentences = []
for image in dataset:
	for sentence in image["sentences"]:
		sentences.append({"id": sentence["sentid"], "raw": " ".join(sentence["tokens"])})
json.dump(sentences, open(args.fout, "w"))

