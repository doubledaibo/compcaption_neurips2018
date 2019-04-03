require 'hdf5'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local t = require 'misc.transforms'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	
	-- load the json file which contains additional information about the dataset
	print('DataLoader loading json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
	self.input_itow = self.info.input_itow
	self.input_wtoi = self.info.input_wtoi
	self.output_itow = self.info.output_itow
	self.output_wtoi = self.info.output_wtoi
	self.input_vocab_size = utils.count_keys(self.input_itow)
	self.output_vocab_size = utils.count_keys(self.output_itow)

	self.batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
	print('input vocab size is ' .. self.input_vocab_size)
	print('output vocab size is ' .. self.output_vocab_size)
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
	local fullseq_size = self.h5_file:read('/fullseqs'):dataspaceSize()
	self.seq_length = fullseq_size[2]
	self.num_fullseq = fullseq_size[1]
	local partseq_size = self.h5_file:read('/partseqs'):dataspaceSize()
	self.num_partseq = partseq_size[1]
	print('max sequence length in data is ' .. self.seq_length)
	-- separate out indexes for each of the provided splits
	self.full_split_ix = {}
	self.full_split_ix["train"] = {}
	self.full_split_ix["val"] = {}
	self.full_split_ix["test"] = {}
	self.full_iterator = {}
	self.full_iterator["train"] = 1
	self.full_iterator["val"] = 1
	self.full_iterator["test"] = 1
	self.full_perm = {}
	local full_splits = self.h5_file:read("/fullseqs_splits"):all()
	for i = 1, self.num_fullseq do
		if full_splits[i] == 1 or full_splits[i] == 3 then
			table.insert(self.full_split_ix["train"], i)
		elseif full_splits[i] == 2 then
			table.insert(self.full_split_ix["val"], i)
		elseif full_splits[i] == 4 then
			table.insert(self.full_split_ix["test"], i)
--		else
--			error("missing splits")
		end
	end

	for k, v in pairs(self.full_split_ix) do
		print(string.format('assigned %d positive samples to split %s', #v, k))
	end

	self.part_split_ix = {}
	self.part_split_ix["train"] = {}
	self.part_split_ix["val"] = {}
	self.part_split_ix["test"] = {}
	self.part_iterator = {}
	self.part_iterator["train"] = 1
	self.part_iterator["val"] = 1
	self.part_iterator["test"] = 1
	self.part_perm = {}
	local part_splits = self.h5_file:read("/partseqs_splits"):all()
	for i = 1, self.num_partseq do
		if part_splits[i] == 1 or part_splits[i] == 3 then
			table.insert(self.part_split_ix["train"], i)
		elseif part_splits[i] == 2 then
			table.insert(self.part_split_ix["val"], i)
		elseif part_splits[i] == 4 then
			table.insert(self.part_split_ix["test"], i)
--		else
--			error("missing splits")
		end
	end

	for k, v in pairs(self.part_split_ix) do
		print(string.format('assigned %d negative samples to split %s', #v, k))
	end


	self.meanstd = {
				mean = { 0.485, 0.456, 0.406 },
				std = { 0.229, 0.224, 0.225 },
			}

	self.transform = t.Compose{
		 t.ColorNormalize(self.meanstd)
	}
end

function DataLoader:init_rand(full, split)
	if full then
		local size = #self.full_split_ix[split]	
		if split == 'train' then
			self.full_perm[split] = torch.randperm(size)
		else
			self.full_perm[split] = torch.range(1,size) -- for test and validation, do not permutate
		end
	else
		local size = #self.part_split_ix[split]
		if split == "train" then
			self.part_perm[split] = torch.randperm(size)
		else
			self.part_perm[split] = torch.range(1, size)
		end
	end
end

function DataLoader:reset_iterator(full, split)
	if full then
		self.full_iterator[split] = 1
	else
		self.part_iterator[split] = 1
	end
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

function DataLoader:getnBatch(split)
	return math.floor((#self.full_split_ix[split] + #self.part_split_ix[split]) / self.batch_size)
end

function DataLoader:run(split)
	local batch_size = self.batch_size
	local seq_length = self.seq_length
	local batch_size = batch_size / 2

	local full_feat_batch = torch.FloatTensor(batch_size, 2048):zero()
	local full_conv_feat_batch = torch.FloatTensor(batch_size, 49, 2048):zero()
	local full_seq_batch = torch.LongTensor(batch_size, seq_length):zero()
	local part_feat_batch = torch.FloatTensor(batch_size, 2048):zero()
	local part_conv_feat_batch = torch.FloatTensor(batch_size, 49, 2048):zero()
	local part_seq_batch = torch.LongTensor(batch_size, seq_length):zero()
	
	if self.full_iterator[split] + batch_size  - 1 > #self.full_split_ix[split] then
		self:reset_iterator(true, split)
		self:init_rand(true, split)
	end

	local indices = self.full_perm[split]:narrow(1, self.full_iterator[split], batch_size)

	for i, ixm in ipairs(indices:totable()) do
		local ix = self.full_split_ix[split][ixm]
		local imgidx = self.h5_file:read("/fullseqs_imgidx"):partial({ix, ix})[1]
		full_feat_batch[i] = self.feat_file:read("/feats"):partial({imgidx, imgidx}, {1, 2048})
		full_conv_feat_batch[i] = self.feat_file:read("/conv_feats"):partial({imgidx, imgidx}, {1, 49}, {1, 2048})
		full_seq_batch[i] = self.h5_file:read("/fullseqs"):partial({ix, ix}, {1, seq_length})
	end

	if self.part_iterator[split] + batch_size  - 1 > #self.part_split_ix[split] then
		self:reset_iterator(false, split)
		self:init_rand(false, split)
	end

	local indices = self.part_perm[split]:narrow(1, self.part_iterator[split], batch_size)

	for i, ixm in ipairs(indices:totable()) do
		local ix = self.part_split_ix[split][ixm]

		local imgidx = self.h5_file:read("/partseqs_imgidx"):partial({ix, ix})[1]
		part_feat_batch[i] = self.feat_file:read("/feats"):partial({imgidx, imgidx}, {1, 2048})
		part_conv_feat_batch[i] = self.feat_file:read("/conv_feats"):partial({imgidx, imgidx}, {1, 49}, {1, 2048})
		part_seq_batch[i] = self.h5_file:read("/partseqs"):partial({ix, ix}, {1, seq_length})
	end

	local batch_data = {}
	batch_data.fullseqs = full_seq_batch:transpose(1,2):contiguous()
	batch_data.partseqs = part_seq_batch:transpose(1,2):contiguous()
	batch_data.fullfeats = full_feat_batch
	batch_data.partfeats = part_feat_batch
	batch_data.fullconv_feats = full_conv_feat_batch
	batch_data.partconv_feats = part_conv_feat_batch

	self.full_iterator[split] = self.full_iterator[split] + batch_size
	self.part_iterator[split] = self.part_iterator[split] + batch_size
	return batch_data
end



