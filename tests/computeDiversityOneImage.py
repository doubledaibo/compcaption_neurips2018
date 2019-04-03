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
sum_avg_dist = 0
n = len(gen_caps)
for k in xrange(n):
	if k % 100 == 0 and k != 0:
		print str(k) + " / " + str(n)
	m = len(gen_caps[k]["caption"])
	assert m == 5
	cnt = 0
	sum_dist = 0
	for i in xrange(m):
		for j in xrange(i + 1, m):
			sum_dist += editdistance.eval(gen_caps[k]["caption"][i].split(" "), gen_caps[k]["caption"][j].split(" "))
			cnt += 1
	sum_avg_dist += (sum_dist + 0.0) / cnt
print sum_avg_dist / n
