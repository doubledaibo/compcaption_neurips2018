require 'nn'
require 'misc.LookupTableMaskZero'
require 'misc.TopDownEncoder'

local net_utils = require 'misc.net_utils'
local utils = require 'misc.utils'

local layer, parent = torch.class('nn.Stoper', 'nn.Module')
function layer:__init(cfg)
	parent.__init(self)

	self.rnn_size = cfg.rnn_size
	self.dropout = cfg.dropout
	self.encoder = nn.TopDownEncoder(cfg)
	self.scorer = nn.Sequential()
		:add(nn.Linear(self.rnn_size, self.rnn_size))
		:add(nn.ReLU())
		:add(nn.Dropout(cfg.dropout))
		:add(nn.Linear(self.rnn_size, 1))
		:add(nn.Sigmoid())
end

function layer:createClones()
	self.encoder:createClones()
end

function layer:shareThinClone(thin_copy)
	self.encoder:shareThinClone(thin_copy.encoder)
	thin_copy.scorer:share(self.scorer, 'weight', 'bias')
end

function layer:getModulesList()
	local modules = {self.scorer}
	local encoder_modules = self.encoder:getModulesList()
	for k, v in pairs(encoder_modules) do table.insert(modules, v) end
	
	return modules
end

function layer:parameters()
	local modules = self:getModulesList()
	local params = {}
	local grad_params = {}
	for dummy, module in pairs(modules) do
		local p, g = module:parameters()
		for k, v in pairs(p) do table.insert(params, v) end
		for k, v in pairs(g) do table.insert(grad_params, v) end
	end
	return params, grad_params
end

function layer:training()
	self.encoder:training()
	self.scorer:training()
end

function layer:evaluate()
	self.encoder:evaluate()
	self.scorer:evaluate()
end

function layer:updateOutput(input)
	self.encode = self.encoder:forward(input)
	self.output = self.scorer:forward(self.encode)
	return self.output	
end

function layer:updateGradInput(input, gradOutput)
	local grad_encode = self.scorer:backward(self.encode, gradOutput)
	self.gradInput = self.encoder:backward(input, grad_encode)
	return self.gradInput
end
