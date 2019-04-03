'''
preparing training data for connecting module
'''
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
parser.add_argument('--fpos', dest='fpos', default='', type=str)
parser.add_argument('--maxl', dest='maxl', default=9, type=int)
parser.add_argument('--thres', dest='thres', default=5, type=int)
parser.add_argument('--imgroot', dest='imgroot', default='../images', type=str)
if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(1)

args = parser.parse_args()

image_annotations = json.load(open(args.fimg))["images"]

f = h5py.File(args.fout, "r+")
nimg = len(image_annotations)
imgs = f.create_dataset("images", (nimg, 3, 256, 256), dtype="uint8")
imgids = f.create_dataset("imageids", (nimg,), dtype="uint32")
idtoidx = {}
senttosplit = {}
senttoid = {}
for i, img_anno in enumerate(image_annotations):
	if i % 100 == 0:
		print str(i) + " / " + str(nimg)
	img = imread(osp.join(args.imgroot, img_anno["filepath"], img_anno["filename"]))
#	img = imread(osp.join(args.imgroot, img_anno["filename"]))
	img = imresize(img, (256, 256))
	if len(img.shape) == 2:
		img = img[:, :, np.newaxis]
		img = np.concatenate((img, img, img), axis = 2)
	img = img.transpose(2, 0, 1)
	imgs[i] = img
	if "cocoid" not in img_anno:
		img_anno["cocoid"] = img_anno["imgid"]
	imgids[i] = img_anno["cocoid"]
	idtoidx[img_anno["cocoid"]] = i + 1
	for sent in img_anno["sentences"]:
		senttosplit[sent["sentid"]] = img_anno["split"]
		senttoid[sent["sentid"]] = img_anno["cocoid"]

counts = {}
for img_anno in image_annotations:
	for sent in img_anno["sentences"]:
		for word in sent["tokens"]:
			counts[word] = counts.get(word, 0) + 1
vocab = [w for w, n in counts.iteritems() if n > args.thres]
vocab.append('UNK')

input_itow = {i + 1: w for i, w in enumerate(vocab)}	
input_wtoi = {w: i + 1 for i, w in enumerate(vocab)}

non_nouns = json.load(open(args.fpos))["others"]
output_vocab = [w for i, w in enumerate(non_nouns) if w in input_wtoi]
output_vocab.append('UNK')
output_itow = {i + 1: w for i, w in enumerate(output_vocab)} 
output_wtoi = {w: i + 1 for i, w in enumerate(output_vocab)}

mappings = {}
mappings["input_itow"] = input_itow
mappings["input_wtoi"] = input_wtoi
mappings["output_itow"] = output_itow
mappings["output_wtoi"] = output_wtoi
json.dump(mappings, open(args.fout + ".mappings.json", "w"))

collected_samples = json.load(open(args.fsent))
collected_samples = [sample for sample in collected_samples if (len(sample["mid"]) > 0) or (len(sample["mid"]) == 0 and len(sample["left"]) > 0 and len(sample["right"]) > 0)]

valid_samples = []
for sample in collected_samples:
	left_words = sample["left"].split(" ") if sample["left"] != "" else []
	right_words = sample["right"].split(" ") if sample["right"] != "" else []
	mid_words = sample["mid"].split(" ") if sample["mid"] != "" else []
	if len(left_words) > args.maxl or len(right_words) > args.maxl or len(mid_words) > args.maxl:
		continue
	for i in xrange(len(left_words)):
		if left_words[i] not in input_wtoi:
			left_words[i] = "UNK"
	for i in xrange(len(right_words)):
		if right_words[i] not in input_wtoi:
			right_words[i] = "UNK"
	for i in xrange(len(mid_words)):
		if mid_words[i] not in input_wtoi:
			mid_words[i] = "UNK"
	valid_samples.append([left_words, right_words, " ".join(mid_words) if len(mid_words) > 0 else "", sample["sentid"]])

counts = {}
for sample in valid_samples:
	counts[sample[2]] = counts.get(sample[2], 0) + 1
validmids = [mid for mid, count in counts.iteritems() if count > args.thres]
nmid = len(validmids)

mseqs = np.zeros((nmid, args.maxl))
midtoidx = {}
for i, mid in enumerate(validmids):
	midtoidx[mid] = i + 1
	if mid == "":
		continue
	for j, word in enumerate(mid.split(" ")):
		mseqs[i][j] = input_wtoi[word]

nsample = 0
for i, sample in enumerate(valid_samples):
	if sample[2] not in midtoidx:
		continue
	nsample += 1
lseqs = np.zeros((nsample, args.maxl))
rseqs = np.zeros((nsample, args.maxl))
#del f["splits"]
splits = f.create_dataset("splits", (nsample, ), dtype="uint32")
#del f["imgidx"]
imgidx = f.create_dataset("imgidx", (nsample, ), dtype="uint32")
#del f["mseqidx"]
mseqidx = f.create_dataset("mseqidx", (nsample, ), dtype="uint32")
idx = 0
for i, sample in enumerate(valid_samples):
	if i % 1000 == 0:
		print str(i) + " / " + str(nsample)
	if sample[2] not in midtoidx:
		continue
	idx += 1
	for j, word in enumerate(sample[0]):
		lseqs[idx - 1][j] = input_wtoi[word]
	for j, word in enumerate(sample[1]):
		rseqs[idx - 1][j] = input_wtoi[word]	
	sentid = sample[-1]
	if senttosplit[sentid] == "train":
		splits[idx - 1] = 1
	elif senttosplit[sentid] == "val":
		splits[idx - 1] = 2
	elif senttosplit[sentid] == "restval":
		splits[idx - 1] = 3
	elif senttosplit[sentid] == "test":
		splits[idx - 1] = 4
	imgidx[idx - 1] = idtoidx[senttoid[sentid]]
	mseqidx[idx - 1] = midtoidx[sample[2]]
#del f["lseqs"]
f.create_dataset("lseqs", data=lseqs, dtype="uint32")
#del f["rseqs"]
f.create_dataset("rseqs", data=rseqs, dtype="uint32")
#del f["mseqs"]
f.create_dataset("mseqs", data=mseqs, dtype="uint32")
f.close()
