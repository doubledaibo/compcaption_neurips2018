require 'torch'
require 'nn'
require 'nngraph'
require 'misc.FeatDataLoaderResNetConnecting'

local utils = require 'misc.utils'
require 'misc.TopDownConnectingModule'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a connecting module')
cmd:text()
cmd:text('Options')


-- Data input settings

cmd:option('-input_h5','data/coco.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_feat', 'data/coco_resnet152_feats.h5', '')
cmd:option('-input_json','data/coco_mappings.json','path to the json file containing additional info and vocab')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-checkpoint_path', 'save/', 'folder to save checkpoints into (empty = this folder)')

cmd:option('-dataset','coco','')

cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-batch_size',20,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')

-- training setting
cmd:option('-nEpochs', 50, 'Max number of training epoch')

--actuall batch size = gpu_num * batch_size

cmd:option('-fc_feat_size',2048,'the encoding size of each token in the vocabulary, and the image.')
--cmd:option('-att_feat_size',2048,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-att_hid_size',512, '')
cmd:option('-att_size', 49, '')

-- Optimization: General
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-dropout', 0.5, 'strength of dropout in the Language Model RNN')

-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 20, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the CNN
--cmd:option('-finetune_start_layer', 6, 'finetune start layer. [1-10]')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', -1, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 5, 'how often to save a model checkpoint?')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '1', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

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


local protos = {}

local loader = DataLoader({feat_file = cfg.input_feat, h5_file = cfg.input_h5, json_file = cfg.input_json, batch_size = cfg.batch_size})

cfg.seq_length = loader:getSeqLength()
cfg.input_vocab_size = loader:getInputVocabSize()
cfg.mid_size = loader:getMidSize() + 1
if cfg.start_from ~= '' then
	print("load checkpoint from " .. cfg.start_from)
	loaded_checkpoint = torch.load(cfg.start_from)
	protos.lm = loaded_checkpoint.protos.lm:cuda()
else
	protos.lm = nn.EmbeddingModel(cfg):cuda()
end
-- criterion for the language model
protos.crit = nn.ClassNLLCriterion():cuda()

params, grad_params = protos.lm:getParameters()

print('total number of parameters in LM: ', params:nElement())

assert(params:nElement() == grad_params:nElement())

protos.thin_lm = protos.lm:clone()
protos.lm:shareThinClone(protos.thin_lm)

protos.lm:createClones()
collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function evaluate_split(split)
	print('=> evaluating ...')
-- setting to the evaluation mode, use only the first gpu
	protos.lm:evaluate()

	local n = 0
	local loss_sum = 0
	local hit = 0
	local total = 0
	local nbatch = loader:getnBatch(split)
	loader:init_rand(split)
	loader:reset_iterator(split)
	for n = 1, nbatch do
		local data = loader:run(split)
--		-- convert the data to cuda
		data.feats = data.feats:cuda()
		data.conv_feats = data.conv_feats:cuda()
		data.lseqs = data.lseqs:cuda()
		data.rseqs = data.rseqs:cuda()
		local pred = protos.lm:forward({data.feats, data.conv_feats, data.lseqs, data.rseqs})
		local dummy, topidx = pred:topk(1, 2, true)
		hit = hit + torch.sum(torch.eq(topidx:long(), data.mseqs))
		data.mseqs = data.mseqs:cuda()
		local loss = protos.crit:forward(pred, data.mseqs)
		loss_sum = loss_sum + loss
		total = total + cfg.batch_size * pred:size(2)
	end
	return loss_sum / nbatch, (hit + 0.0) / total
end

local function TrainLM(epoch, opt)

	protos.lm:training()

	grad_params:zero()

	local data = loader:run('train')
	-- convert the data to cuda
	data.feats = data.feats:cuda()
	data.conv_feats = data.conv_feats:cuda()
	data.lseqs = data.lseqs:cuda()
	data.rseqs = data.rseqs:cuda()
	data.mseqs = data.mseqs:cuda()
	local pred, grad_pred
	local loss = 0
	pred = protos.lm:forward({data.feats, data.conv_feats, data.lseqs, data.rseqs})
	loss = loss + protos.crit:forward(pred, data.mseqs)
	grad_pred = protos.crit:backward(pred, data.mseqs)
	protos.lm:backward({data.feats, data.conv_feats, data.lseqs, data.rseqs}, grad_pred)
	grad_params:clamp(-cfg.grad_clip, cfg.grad_clip)
	
	-- update the parameters
	if cfg.optim == 'rmsprop' then
		rmsprop(params, grad_params, learning_rate, cfg.optim_alpha, cfg.optim_epsilon, optim_state)
	elseif cfg.optim == 'adam' then
		adam(params, grad_params, learning_rate, cfg.optim_alpha, cfg.optim_beta, cfg.optim_epsilon, optim_state)
	else
		error('bad option cfg.optim')
	end

	return loss
end

paths.mkdir(cfg.checkpoint_path)

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
optim_state = {}
learning_rate = cfg.learning_rate
iter = 0
epoch = 0

nbatch = loader:getnBatch('train')

local checkpoint_path = path.join(cfg.checkpoint_path, 'model_' .. cfg.id)
local timer = torch.Timer()
--evaluate_split('val')	
iter = 0
epoch = 0
while true do
	iter = iter + 1
	if iter % nbatch == 1 then
		loader:init_rand('train')
		loader:reset_iterator('train')
		epoch = epoch + 1
		if epoch > cfg.nEpochs then break end
	end
	local loss = 0
	loss = TrainLM(epoch, opt)
	if iter % 10 == 0 then
		print('lm_learning_rate: ' .. learning_rate)
		print("iter: " .. iter .. " / " .. nbatch .. " epoch: " .. epoch .. ", loss: " .. loss)
		collectgarbage() 
	end
		
	if iter % cfg.save_checkpoint_every == 0 then
		local val_loss, val_acc = evaluate_split('val')	
		print('validation loss for # ' .. epoch .. ' : ' .. val_loss .. ' , ' .. val_acc)

		local checkpoint = {}
		local save_protos = {}
		save_protos.lm = protos.thin_lm
			
		checkpoint.protos = save_protos
		torch.save(checkpoint_path .. '_iter' .. iter .. '.t7', checkpoint)
		print('wrote checkpoint to ' .. checkpoint_path .. '_iter' .. iter .. '.t7')
--		aaa:clone()
	end
end


