require 'hdf5'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local t = require 'misc.transforms'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	
	-- load the json file which contains additional information about the dataset
	print('Loading vocab mapping file: ', opt.vocab_mapping)
	self.vocab_mapping = utils.read_json(opt.vocab_mapping)
	self.input_itow = self.vocab_mapping.input_itow
	self.input_wtoi = self.vocab_mapping.input_wtoi
	self.output_itow = self.vocab_mapping.output_itow
	self.output_wtoi = self.vocab_mapping.output_wtoi
	self.input_vocab_size = utils.count_keys(self.input_itow)
	self.output_vocab_size = utils.count_keys(self.output_itow)

	self.batch_size = 1
	print('input vocab size is ' .. self.input_vocab_size)
	print('output vocab size is ' .. self.output_vocab_size)
	print('Loading combination order file: ', opt.combination_order)
	self.combination_order = utils.read_json(opt.combination_order)
	-- open the hdf5 files
	print('Loading imgfeat h5 file: ', opt.imgfeat_h5)
	self.feat_h5 = hdf5.open(opt.imgfeat_h5)
	print('Loading nounphrase h5 file: ', opt.nounphrase_h5)
	self.nounphrase_h5 = hdf5.open(opt.nounphrase_h5, 'r')
	self.nounphrasepred_h5 = hdf5.open(opt.nounphrasepred_h5, 'r')
	local nounphrase_size = self.nounphrase_h5:read("/nouns"):dataspaceSize()
	self.nounphrase_vocab_size = nounphrase_size[2]
	self.nounphraseencode_h5 = hdf5.open(opt.nounphrase_encode_h5, 'r')
	local encode_size = self.nounphraseencode_h5:read("/encode"):dataspaceSize()
	self.encode_length = encode_size[2]
	print('Loading combination h5 file: ', opt.combination_h5)
	self.combination_h5 = hdf5.open(opt.combination_h5, 'r')	

	-- extract image size from dataset
	local images_size = self.nounphrase_h5:read('/images'):dataspaceSize()
	self.num_images = images_size[1]

	print(string.format('read %d images', self.num_images))
	-- separate out indexes for each of the provided splits
	
	local combination_size = self.combination_h5:read("/lseqs"):dataspaceSize()
	self.combination_length = combination_size[2]
	local mseq_size = self.combination_h5:read("/mseqs"):dataspaceSize()
	self.num_mseq = mseq_size[1]

	self.split_ix = {}
	self.split_ix["train"] = {}
	self.split_ix["val"] = {}
	self.split_ix["test"] = {}
	self.iterator = {}
	self.iterator["train"] = 1
	self.iterator["val"] = 1
	self.iterator["test"] = 1
	self.perm = {}
	local splits = self.nounphrase_h5:read("/splits"):all()
	for i = 1, self.num_images do
		if splits[i] == 1 or splits[i] == 3 then
			table.insert(self.split_ix["train"], i)
		elseif splits[i] == 2 then
			table.insert(self.split_ix["val"], i)
		elseif splits[i] == 4 then
			table.insert(self.split_ix["test"], i)
		else
			print("missing splits")
		end
	end
	local splits = nil
	collectgarbage()
	for k, v in pairs(self.split_ix) do
		print(string.format('assigned %d images to split %s', #v, k))
	end
	
	self:_prepareNounPhraseIdtoIdx()
end

function DataLoader:_prepareNounPhraseIdtoIdx()
	self.nounphrase_idtoidx = {}
	local imgids = self.nounphrase_h5:read("/imageids"):all()
	for i = 1, imgids:size(1) do
		self.nounphrase_idtoidx[imgids[i]] = i
	end 		
end

function DataLoader:init_rand(split)
	local size = #self.split_ix[split]	
	if split == 'train' then
		self.perm[split] = torch.randperm(size)
	else
		self.perm[split] = torch.range(1,size) -- for test and validation, do not permutate
	end
end

function DataLoader:reset_iterator(split)
	self.iterator[split] = 1
end

function DataLoader:getInputVocabSize()
	return self.input_vocab_size
end

function DataLoader:getOutputVocabSize()
	return self.output_vocab_size
end

function DataLoader:getInputItoW()
	return self.input_itow
end

function DataLoader:getInputWtoI()
	return self.input_wtoi
end

function DataLoader:getOutputItoW()
	return self.output_itow
end

function DataLoader:getOutputWtoI()
	return self.output_wtoi
end

function DataLoader:getNumMSeq()
	return self.num_mseq
end

function DataLoader:getnBatch(split)
	return math.floor(#self.split_ix[split] / self.batch_size)
end

function DataLoader:getNounPhraseEncode(idx)
	local encode = self.nounphraseencode_h5:read("/encode"):partial({idx, idx}, {1, self.encode_length}) 
	return encode:transpose(1, 2):contiguous()
end

function DataLoader:getNounPhrasePreds(img_id)
	local idx = self.nounphrase_idtoidx[img_id]
	local pred = self.nounphrasepred_h5:read("/pred"):partial({idx, idx}, {1, self.nounphrase_vocab_size})
	return pred
end

function DataLoader:getGTNounPhrasePreds(img_id)
	local idx = self.nounphrase_idtoidx[img_id]
	local pred = self.nounphrase_h5:read("/nouns"):partial({idx, idx}, {1, self.nounphrase_vocab_size})
	return pred		
end

function DataLoader:getGTNounPhrases(img_id)
	local orders = self.combination_order[tostring(img_id)]
	local subphrases = {}
	for k, order in pairs(orders) do
		for i = 1, #order do 
			if string.find(order[i].idf, "+") == nil then
				local phrase = torch.LongTensor(1, self.combination_length):zero()
				local st = 1
				local en = #order[i].phrase
				if en > self.combination_length then st = en - self.combination_length + 1 end
				for j = st, en do
					if self.input_wtoi[order[i].phrase[j]] == nil then
						phrase[1][j - st + 1] = self.input_wtoi["UNK"]
					else
						phrase[1][j - st + 1] = self.input_wtoi[order[i].phrase[j]]
					end
				end
				local subphrase = {}
				subphrase.lp = 0
				subphrase.phrase = phrase:transpose(1, 2):contiguous()
				table.insert(subphrases, subphrase)
			end
		end
	end
	return subphrases
end

function DataLoader:getGTOrders(img_id)
	local orders = self.combination_order[tostring(img_id)]
	local choice = math.min(#orders, math.floor(torch.rand(1)[1] * #orders) + 1)
	local order = orders[choice]  --randomly select a combination order
	local subphrases = {}
	local idfs = {}
	for i = 1, #order do
		if string.find(order[i].idf, "+") == nil then
			local phrase = torch.LongTensor(1, self.combination_length):zero()
			local st = 1
			local en = #order[i].phrase
			if en > self.combination_length then st = en - self.combination_length + 1 end
			for j = st, en do
				if self.input_wtoi[order[i].phrase[j]] == nil then
					phrase[1][j - st + 1] = self.input_wtoi["UNK"]
				else
					phrase[1][j - st + 1] = self.input_wtoi[order[i].phrase[j]]
				end
			end
			local subphrase = {}
			subphrase.lp = 0
			subphrase.phrase = phrase:transpose(1, 2):contiguous()
			subphrase.idf = order[i].idf
			table.insert(subphrases, subphrase)
		else
			table.insert(idfs, order[i].idf)
		end
	end
	return subphrases, idfs
end

function DataLoader:nextImg(split)
	local size, batch_size = #self.split_ix[split], self.batch_size
	local split_ix = self.split_ix[split]
	if self.iterator[split] > size then
		self:reset_iterator(split)
		self:init_rand(split)
	end
	local ix = split_ix[self.perm[split][self.iterator[split]]]
	self.iterator[split] = self.iterator[split] + 1
	local img_feat = torch.FloatTensor(1, 2048):zero()
	img_feat[1] = self.feat_h5:read("/feats"):partial({ix, ix}, {1, 2048})
	local img_id = self.nounphrase_h5:read("/imageids"):partial({ix, ix})[1]
	return img_feat, img_id
end

function DataLoader:nextImgConv(split)
	local size, batch_size = #self.split_ix[split], self.batch_size
	local split_ix = self.split_ix[split]
	if self.iterator[split] > size then
		self:reset_iterator(split)
		self:init_rand(split)
	end
	local ix = split_ix[self.perm[split][self.iterator[split]]]
	self.iterator[split] = self.iterator[split] + 1
	local img_feat = torch.FloatTensor(1, 2048):zero()
	local conv_feat = torch.FloatTensor(1, 49, 2048):zero()
	img_feat[1] = self.feat_h5:read("/feats"):partial({ix, ix}, {1, 2048})
	conv_feat[1] = self.feat_h5:read("/conv_feats"):partial({ix, ix}, {1, 49}, {1, 2048})
	local img_id = self.nounphrase_h5:read("/imageids"):partial({ix, ix})[1]
	return img_feat, conv_feat, img_id
end


function DataLoader:getMSeq(st, en)
	local mseq = self.combination_h5:read("/mseqs"):partial({st, en}, {1, self.combination_length}):clone()
	return mseq:transpose(1, 2):contiguous()
end



