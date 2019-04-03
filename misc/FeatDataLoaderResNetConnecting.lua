require 'hdf5'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local t = require 'misc.transforms'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	
	print('DataLoader loading json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
	self.input_itow = self.info.input_itow
	self.input_wtoi = self.info.input_wtoi
	self.input_vocab_size = utils.count_keys(self.input_itow)

	self.batch_size = utils.getopt(opt, 'batch_size', 5) 
	print('input vocab size is ' .. self.input_vocab_size)
	-- open the hdf5 file
	print('DataLoader loading h5 file: ', opt.h5_file)
	self.h5_file = hdf5.open(opt.h5_file, 'r')
	print('DataLoader loading feat file: ', opt.feat_file)
	self.feat_file = hdf5.open(opt.feat_file, 'r')
 
	-- extract image size from dataset
	local images_size = self.h5_file:read('/images'):dataspaceSize()
	assert(#images_size == 4, '/images should be a 4D tensor')
	assert(images_size[3] == images_size[4], 'width and height must match')
	self.num_images = images_size[1]
--	self.num_channels = images_size[2]
--	self.max_image_size = images_size[3]

	print(string.format('read %d images', self.num_images))
	-- load in the sequence data
	local seq_size = self.h5_file:read('/lseqs'):dataspaceSize()
	self.seq_length = seq_size[2]
	self.num_seq = seq_size[1]
	print('max sequence length in data is ' .. self.seq_length)
	local mseq_size = self.h5_file:read("/mseqs"):dataspaceSize()
	self.num_mseq = mseq_size[1]
	-- separate out indexes for each of the provided splits
	self.split_ix = {}
	self.split_ix["train"] = {}
	self.split_ix["val"] = {}
	self.split_ix["test"] = {}
	self.iterator = {}
	self.iterator["train"] = 1
	self.iterator["val"] = 1
	self.iterator["test"] = 1
	self.perm = {}
	local splits = self.h5_file:read("/splits"):all()
	for i = 1, self.num_seq do
		if splits[i] == 1 or splits[i] == 3 then
			table.insert(self.split_ix["train"], i)
		elseif splits[i] == 2 then
			table.insert(self.split_ix["val"], i)
		elseif splits[i] == 4 then
			table.insert(self.split_ix["test"], i)
		--else 
		--	error("missing splits")
		end
	end

	for k, v in pairs(self.split_ix) do
		print(string.format('assigned %d images to split %s', #v, k))
	end

	self.meanstd = {
				mean = { 0.485, 0.456, 0.406 },
				std = { 0.229, 0.224, 0.225 },
			}

	self.transform = t.Compose{
		 t.ColorNormalize(self.meanstd)
	}
	
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

function DataLoader:getInputItoW()
	return self.input_itow
end

function DataLoader:getInputWtoI()
	return self.input_wtoi
end

function DataLoader:getSeqLength()
	return self.seq_length
end

function DataLoader:getMidSize()
	return self.num_mseq
end

function DataLoader:getnBatch(split)
	return math.floor(#self.split_ix[split] / self.batch_size)
end

function DataLoader:getMSeq(st, en)
	local mseq = self.h5_file:read("/mseqs"):partial({st, en}, {1, self.seq_length}):clone()
	return mseq:transpose(1, 2):contiguous()
end

function DataLoader:run(split)
	local size, batch_size = #self.split_ix[split], self.batch_size
	local seq_length = self.seq_length

	local split_ix = self.split_ix[split]
	
	if self.iterator[split] + batch_size - 1 > size then
		self:reset_iterator(split)
		self:init_rand(split)
	end

	local indices = self.perm[split]:narrow(1, self.iterator[split], batch_size)

	local feat_batch, conv_feat_batch, lseq_batch, rseq_batch, mseq_batch
	feat_batch = torch.FloatTensor(batch_size * 2, 2048):zero()
	conv_feat_batch = torch.FloatTensor(batch_size * 2, 49, 2048):zero()
	lseq_batch = torch.LongTensor(batch_size * 2, seq_length):zero()
	rseq_batch = torch.LongTensor(batch_size * 2, seq_length):zero()
	mseq_batch = torch.LongTensor(batch_size * 2):zero()
	local img_ids = {}
	local img_idces = {}
	for i, ixm in ipairs(indices:totable()) do
		local ix = split_ix[ixm]

		local imgidx = self.h5_file:read("/imgidx"):partial({ix, ix})[1]
		local imgid = self.h5_file:read("/imageids"):partial({imgidx, imgidx})[1]	
		table.insert(img_idces, imgidx)
		table.insert(img_ids, imgid)
		feat_batch[i] = self.feat_file:read("/feats"):partial({imgidx, imgidx}, {1, 2048})
		conv_feat_batch[i] = self.feat_file:read("/conv_feats"):partial({imgidx, imgidx}, {1, 49}, {1, 2048})
		lseq_batch[i] = self.h5_file:read("/lseqs"):partial({ix, ix}, {1, seq_length})
		rseq_batch[i] = self.h5_file:read("/rseqs"):partial({ix, ix}, {1, seq_length})
		local midx = self.h5_file:read("/mseqidx"):partial({ix, ix})[1]
		mseq_batch[i] = midx
		local lidx = math.floor(torch.rand(1)[1] * (size - 1)) + 1
		if lidx == size then lidx = size - 1 end
		if lidx >= ixm then lidx = split_ix[lidx + 1] else lidx = split_ix[lidx] end
		feat_batch[batch_size + i] = feat_batch[i]
		conv_feat_batch[batch_size + i] = conv_feat_batch[i]
		lseq_batch[batch_size + i] = self.h5_file:read("/lseqs"):partial({lidx, lidx}, {1, seq_length})
		rseq_batch[batch_size + i] = rseq_batch[i]
		mseq_batch[batch_size + i] = self.num_mseq + 1
	end
	local batch_data = {}
	batch_data.lseqs = lseq_batch:transpose(1,2):contiguous()
	batch_data.rseqs = rseq_batch:transpose(1,2):contiguous()
	batch_data.mseqs = mseq_batch
	batch_data.feats = feat_batch
	batch_data.conv_feats = conv_feat_batch
	batch_data.img_id = img_ids
	batch_data.img_idces = img_idces
	
	self.iterator[split] = self.iterator[split] + batch_size
	return batch_data
end



