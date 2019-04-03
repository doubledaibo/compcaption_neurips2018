require 'hdf5'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local t = require 'misc.transforms'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	
	-- load the json file which contains additional information about the dataset
	print('DataLoader loading json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
	self.noun_itow = self.info.noun_itow
	self.noun_wtoi = self.info.noun_wtoi
	self.vocab_size = utils.count_keys(self.noun_itow)
	
	self.batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
	print('vocab size is ' .. self.vocab_size)
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
	for i = 1, self.num_images do
		if splits[i] == 1 or splits[i] == 3 then
			table.insert(self.split_ix["train"], i)
		elseif splits[i] == 2 then
			table.insert(self.split_ix["val"], i)
		elseif splits[i] == 4 then
			table.insert(self.split_ix["test"], i)
		else
			error("missing splits")
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

function DataLoader:getVocabSize()
	return self.vocab_size
end

function DataLoader:getNounItoW()
	return self.noun_itow
end

function DataLoader:getnBatch(split)
	return math.floor(#self.split_ix[split] / self.batch_size)
end

function DataLoader:run(split)
	local size, batch_size = #self.split_ix[split], self.batch_size

	local split_ix = self.split_ix[split]
	
	if self.iterator[split] + batch_size - 1 > size then
		self:reset_iterator(split)
		self:init_rand(split)
	end

	local indices = self.perm[split]:narrow(1, self.iterator[split], batch_size)

	local feat_batch, conv_feat_batch, label_batch
	feat_batch = torch.FloatTensor(batch_size, 2048):zero()
	label_batch = torch.LongTensor(batch_size, self.vocab_size):zero()
	local img_ids = {}
	for i, ixm in ipairs(indices:totable()) do
		local ix = split_ix[ixm]

		local imgid = self.h5_file:read("/imageids"):partial({ix, ix})[1]	
		table.insert(img_ids, imgid)
		feat_batch[i] = self.feat_file:read("/feats"):partial({ix, ix}, {1, 2048})
		label_batch[i] = self.h5_file:read("/nouns"):partial({ix, ix}, {1, self.vocab_size})
	end
	local batch_data = {}
	batch_data.labels = label_batch
	batch_data.feats = feat_batch
	batch_data.img_id = img_ids
	self.iterator[split] = self.iterator[split] + batch_size
	return batch_data
end



