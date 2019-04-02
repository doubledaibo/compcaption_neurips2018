'''
preparing training data for nounphrase prediction
'''
import argparse
import sys
import json
import h5py
import numpy as np
from scipy.misc import imread, imresize
from os import path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--fnounphrase', dest='fnounphrase', default='', type=str)
parser.add_argument('--fimg', dest='fimg', default='', type=str)
parser.add_argument('--fout', dest='fout', default='', type=str)
parser.add_argument('--fmapping', dest='fmapping', default='', type=str)
parser.add_argument('--maxl', dest='maxl', default=2, type=int)
parser.add_argument('--thres', dest='thres', default=3, type=int)
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
#del f["splits"]
splits = f.create_dataset("splits", (nimg,), dtype="uint32")
idtoidx = {}
senttosplit = {}
senttoid = {}
for i, img_anno in enumerate(image_annotations):
	if i % 10000 == 0:
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
	if img_anno["split"] == "train":
		splits[i] = 1 
	elif img_anno["split"] == "val":
		splits[i] = 2
	elif img_anno["split"] == "restval":
		splits[i] = 3
	elif img_anno["split"] == "test":
		splits[i] = 4
	idtoidx[img_anno["cocoid"]] = i
	for sent in img_anno["sentences"]:
		senttosplit[sent["sentid"]] = img_anno["split"]
		senttoid[sent["sentid"]] = img_anno["cocoid"]
	if i < 10:
		print img_anno["cocoid"]	

input_wtoi = json.load(open(args.fmapping))["input_wtoi"]
nounphrases = json.load(open(args.fnounphrase))
valid_nounphrase = {}
all_nounphrases = []
for nounphrase in nounphrases:
	if nounphrase["right"] not in input_wtoi:
		continue
	nounphrase["all"] = nounphrase["mid"] + " " + nounphrase["right"] if nounphrase["mid"] != "" else nounphrase["right"]
	if len(nounphrase["all"].split(" ")) > args.maxl:
		continue
	n = 0
	words = nounphrase["all"].split(" ")
	for word in words:
		if word not in input_wtoi:
			n += 1
	if n > 0:
		continue
	valid_nounphrase[nounphrase["all"]] = valid_nounphrase.get(nounphrase["all"], 0) + 1
	all_nounphrases.append(nounphrase)
	#	words = noun.split(" ")
	#	for i in xrange(len(words)):
	#		if words[i] not in wtoi:
	#			words[i] = "UNK"
	#	noun = " ".join(words)
	#	valid_nouns[noun] = 1


tmp = [w for w, n in valid_nounphrase.iteritems() if n > args.thres]
nnounphrase = len(tmp)
nounphrase_wtoi = {w : i + 1 for i, w in enumerate(tmp)}
nounphrase_itow = {i + 1 : w for i, w in enumerate(tmp)}

mappings = {}
mappings["noun_itow"] = nounphrase_itow
mappings["noun_wtoi"] = nounphrase_wtoi
json.dump(mappings, open(args.fout + ".mappings.json", "w"))

nounphrase_gt = np.zeros((nimg, nnounphrase))

for nounphrase in all_nounphrases:
	idx = idtoidx[senttoid[nounphrase["sentid"]]]
	if nounphrase["all"] in nounphrase_wtoi:
		nounphrase_gt[idx][nounphrase_wtoi[nounphrase["all"]] - 1] = 1
#del f["nouns"]
f.create_dataset("nouns", data=nounphrase_gt, dtype="uint8")
f.close()
