import argparse
import sys
import json
import editdistance

parser = argparse.ArgumentParser()
parser.add_argument('--fin', dest='fin', default='', type=str)

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(1)

args = parser.parse_args()

gen_caps = json.load(open(args.fin))
cnt = 0
sum_dist = 0
n = len(gen_caps)
for i in xrange(n):
	if i % 100 == 0 and i != 0:
		print str(i) + " / " + str(n)
	for j in xrange(i + 1, n):
		sum_dist += editdistance.eval(gen_caps[i]["caption"].split(" "), gen_caps[j]["caption"].split(" "))
		cnt += 1
print (sum_dist + 0.0) / cnt
