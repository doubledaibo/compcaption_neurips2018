import argparse
import sys
import json
import h5py
import numpy as np
from scipy.misc import imread, imresize
from os import path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--fsent', dest='fsent', default='', type=str)
parser.add_argument('--fimg', dest='fimg', default='', type=str)
parser.add_argument('--fout', dest='fout', default='', type=str)
parser.add_argument('--fmapping', dest='fmapping', default='', type=str)

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(1)

args = parser.parse_args()

image_annotations = json.load(open(args.fimg))["images"]

nimg = len(image_annotations)
senttoid = {}
for i, img_anno in enumerate(image_annotations):
	if i % 100 == 0:
		print str(i) + " / " + str(nimg)
	for sent in img_anno["sentences"]:
		senttoid[sent["sentid"]] = img_anno["cocoid"]

orders = json.load(open(args.fsent))
ordermapping = {}
for sample in orders:
	imgid = senttoid[sample["sentid"]]
	if imgid not in ordermapping:
		ordermapping[imgid] = []
	n = len(sample["order"])
	for i in xrange(n):
		sample["order"][i]["phrase"] = sample["order"][i]["phrase"].split(" ")
	ordermapping[imgid].append(sample["order"])

json.dump(ordermapping, open(args.fout, "w"))
