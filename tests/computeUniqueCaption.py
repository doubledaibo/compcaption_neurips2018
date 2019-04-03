import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--fin', dest='fin', default='', type=str)

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(1)

args = parser.parse_args()

gen_caps = json.load(open(args.fin))
unique = {}
for cap in gen_caps:
	if cap["caption"] not in unique:
		unique[cap["caption"]] = 1
print (len(unique.keys()) + 0.0) / len(gen_caps)
