import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--fmapping', dest='fmapping', default='', type=str)
parser.add_argument('--fcap', dest='fcap', default='', type=str)

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(1)

args = parser.parse_args()

input_wtoi = json.load(open(args.fmapping))["input_wtoi"]
caps = json.load(open(args.fcap))

n = len(caps)
uses = {}
for cap in caps:
	for word in cap["caption"].split(" "):
		if word in input_wtoi:
			uses[word] = 1
m = len(uses.keys())
print m
print (m + 0.0) / len(input_wtoi.keys())
