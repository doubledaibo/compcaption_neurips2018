require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'

local utils = require 'misc.utils'

local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Eval')
cmd:text()
cmd:text('Options')


-- Data input settings
cmd:option('-imgfeat_h5', '', '')
cmd:option('-nounphrase_h5', '', '')
cmd:option('-nounphraser', '', '')
cmd:option('-target_h5', 'tmp.h5', '')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-id', "1", '')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local cfg = cmd:parse(arg)
--torch.manualSeed(cfg.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if cfg.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	if cfg.backend == 'cudnn' then require 'cudnn' end
	--cutorch.manualSeed(cfg.seed)
	cutorch.setDevice(cfg.gpuid + 1) -- note +1 because lua is 1-indexed
end

cfg.batch_size = 1

local protos = {}

if cfg.nounphraser ~= '' then
	print("load nounphraser from " .. cfg.nounphraser)
	local loaded_checkpoint = torch.load(cfg.nounphraser)
	protos.nounphraser = loaded_checkpoint.predictor:cuda()
else
	error("need a nounphraser")
end

protos.nounphraser:evaluate()

local h5_file = hdf5.open(cfg.nounphrase_h5, "r")
local feat_file = hdf5.open(cfg.imgfeat_h5, "r")

label_size = h5_file:read("/nouns"):dataspaceSize()
n = label_size[1]
m = label_size[2]
local pred = torch.FloatTensor(n, m):zero()

for i = 1, n do
	if i % 2000 == 0 then
		print(i .. " / " .. n)
	end
	local img_feat = feat_file:read("feats"):partial({i, i}, {1, 2048}):cuda()
	pred[i] = protos.nounphraser:forward(img_feat):float()
end

local cacheFile = hdf5.open(cfg.target_h5, "w")
cacheFile:write("/pred", pred)
cacheFile:close()
