'''
preparing training data for evaluation module
'''
import argparse
import sys
import json
import h5py
import numpy as np
from scipy.misc import imread, imresize
from os import path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--fcombineseq', dest='fcombine', default='', type=str)
parser.add_argument('--fnounseq', dest='fnoun', default='', type=str)
parser.add_argument('--fimg', dest='fimg', default='', type=str)
parser.add_argument('--fout', dest='fout', default='', type=str)
parser.add_argument('--fmapping', dest='fmapping', default='', type=str)
parser.add_argument('--imgroot', dest='imgroot', default='images', type=str)

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
idtofull = {}
for i, img_anno in enumerate(image_annotations):
	if i % 100 == 0:
		print str(i) + " / " + str(nimg)
	img = imread(osp.join(args.imgroot, img_anno["filepath"], img_anno["filename"]))
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
	idtofull[img_anno["cocoid"]] = []
	for sent in img_anno["sentences"]:
		idtofull[img_anno["cocoid"]].append(" ".join(sent["tokens"]))
	
mappings = json.load(open(args.fmapping))
input_itow = mappings["input_itow"]
input_wtoi = mappings["input_wtoi"]

combine_samples = json.load(open(args.fcombine))
noun_samples = json.load(open(args.fnoun))
collected_samples = combine_samples + noun_samples
#collected_samples = [sample for sample in collected_samples if len(sample["mid"]) > 0]

full_sentences = []
part_sentences = []
for sample in collected_samples:
	imgid = senttoid[sample["sentid"]]
	cur_sent = sample["left"]
	if sample["mid"] != "":
		if cur_sent != "":
			cur_sent += " "
		cur_sent += sample["mid"]
	if sample["right"] != "":
		if cur_sent != "":
			cur_sent += " "
		cur_sent += sample["right"]
	if cur_sent == "" or len(cur_sent.split(" ")) > 18:
		continue
	full = False
	for sent in idtofull[imgid]:
		if sent == cur_sent:
			full_sentences.append({"sentid": sample["sentid"], "sent": cur_sent})
			full = True
			break
	if not full:
		part_sentences.append({"sentid": sample["sentid"], "sent": cur_sent})
			 

nfullsent = len(full_sentences)
npartseqs = len(part_sentences)

fullseqs = np.zeros((nfullsent, 18))
partseqs = np.zeros((npartseqs, 18))
#del f["splits"]
#del f["fullseqs_splits"]
#del f["partseqs_splits"]
fullseqs_splits = f.create_dataset("fullseqs_splits", (nfullsent, ), dtype="uint32")
partseqs_splits = f.create_dataset("partseqs_splits", (npartseqs, ), dtype="uint32")
#del f["imgidx"]
#del f["fullseqs_imgidx"]
#del f["partseqs_imgidx"]
fullseqs_imgidx = f.create_dataset("fullseqs_imgidx", (nfullsent, ), dtype="uint32")
partseqs_imgidx = f.create_dataset("partseqs_imgidx", (npartseqs, ), dtype="uint32")

for i, sample in enumerate(full_sentences):
	if i % 1000 == 0:
		print str(i) + " / " + str(nfullsent)
	for j, word in enumerate(sample["sent"].split(" ")):
		fullseqs[i][j] = input_wtoi[word] if word in input_wtoi else input_wtoi["UNK"]
	
	sentid = sample["sentid"]
	if senttosplit[sentid] == "train":
		fullseqs_splits[i] = 1
	elif senttosplit[sentid] == "val":
		fullseqs_splits[i] = 2
	elif senttosplit[sentid] == "restval":
		fullseqs_splits[i] = 3
	elif senttosplit[sentid] == "test":
		fullseqs_splits[i] = 4
	fullseqs_imgidx[i] = idtoidx[senttoid[sentid]]

for i, sample in enumerate(part_sentences):
	if i % 1000 == 0:
		print str(i) + " / " + str(npartseqs)
	for j, word in enumerate(sample["sent"].split(" ")):
		partseqs[i][j] = input_wtoi[word] if word in input_wtoi else input_wtoi["UNK"]
	
	sentid = sample["sentid"]
	if senttosplit[sentid] == "train":
		partseqs_splits[i] = 1
	elif senttosplit[sentid] == "val":
		partseqs_splits[i] = 2
	elif senttosplit[sentid] == "restval":
		partseqs_splits[i] = 3
	elif senttosplit[sentid] == "test":
		partseqs_splits[i] = 4
	partseqs_imgidx[i] = idtoidx[senttoid[sentid]]

#del f["lseqs"]
#del f["mseqs"]
#del f["mseqidx"]
#del f["rseqs"]
#del f["fullseqs"]
#del f["partseqs"]
f.create_dataset("fullseqs", data=fullseqs, dtype="uint32")
f.create_dataset("partseqs", data=partseqs, dtype="uint32")
f.close()
