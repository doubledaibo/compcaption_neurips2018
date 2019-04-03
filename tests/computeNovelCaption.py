import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--fin', dest='fin', default='', type=str)
parser.add_argument('--dataset', dest='dataset', default='dataset_coco.json', type=str)

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(1)

args = parser.parse_args()


anno = json.load(open(args.dataset))["images"]

training_caps = {}

for img in anno:
	if img["split"] == "train" or img["split"] == "restval":
		for sent in img["sentences"]:
			training_caps[" ".join(sent["tokens"])] = 1

gen_caps = json.load(open(args.fin))
n = 0
for cap in gen_caps:
	if cap["caption"] not in training_caps:
		n += 1
print (n + 0.0) / len(gen_caps)
