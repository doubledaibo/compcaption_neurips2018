require 'nn'
require 'misc.LookupTableMaskZero'
require 'misc.TopDownEncoder'

local net_utils = require 'misc.net_utils'
local utils = require 'misc.utils'

local layer, parent = torch.class('nn.EmbeddingModel', 'nn.Module')
function layer:__init(cfg)
	parent.__init(self)

	self.encode_size = cfg.rnn_size
	self.dropout = cfg.dropout
	self.mid_size = cfg.mid_size
	self.leftencoder = nn.TopDownEncoder(cfg)
	self.leftencoder_dropout = nn.Dropout(self.dropout)
--	self.rightencoder = nn.TopDownEncoder(cfg)
	self.rightencoder_dropout = nn.Dropout(self.dropout)
	self.lrcombiner = nn.Sequential()
			:add(nn.JoinTable(1, 1))
			:add(nn.Linear(self.encode_size * 2, self.encode_size))
			:add(nn.ReLU())
			:add(nn.Dropout(self.dropout))
			:add(nn.Linear(self.encode_size, self.encode_size))
			:add(nn.ReLU())
			:add(nn.Dropout(self.dropout))
			:add(nn.Linear(self.encode_size, self.mid_size))
			:add(nn.LogSoftMax())
end

function layer:createClones()
	self.rightencoder = self.leftencoder:clone()
	self.leftencoder:shareThinClone(self.rightencoder)
	self.leftencoder:createClones()
	self.rightencoder:createClones()
end

function layer:shareThinClone(thin_copy)
	self.leftencoder:shareThinClone(thin_copy.leftencoder)
	thin_copy.lrcombiner:share(self.lrcombiner, 'weight', 'bias')
end

function layer:getModulesList()
	local modules = {self.lrcombiner, self.scorer}
	local encoder_modules = self.leftencoder:getModulesList()
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
	self.leftencoder:training()
	self.rightencoder:training()
	self.lrcombiner:training()
	self.leftencoder_dropout:training()
	self.rightencoder_dropout:training()
end

function layer:evaluate()
	self.leftencoder:evaluate()
	self.rightencoder:evaluate()
	self.lrcombiner:evaluate()
	self.leftencoder_dropout:evaluate()
	self.rightencoder_dropout:evaluate()
end

function layer:updateOutput(input)
	local img = input[1]
	local att = input[2]
	local left_seq = input[3]
	local right_seq = input[4]
	self.left_encode = self.leftencoder:forward({img, att, left_seq})
	self.left_encode_dropout = self.leftencoder_dropout:forward(self.left_encode)	
	self.right_encode = self.rightencoder:forward({img, att, right_seq})
	self.right_encode_dropout = self.rightencoder_dropout:forward(self.right_encode)
	self.output = self.lrcombiner:forward({self.left_encode_dropout, self.right_encode_dropout})
	return self.output	
end

function layer:updateGradInput(input, gradOutput)
	local grad_lrcombine_inputs = self.lrcombiner:backward({self.left_encode_dropout, self.right_encode_dropout}, gradOutput)
	local grad_left_inputs = self.leftencoder:backward({input[1], input[2], input[3]}, self.leftencoder_dropout:backward(self.left_encode, grad_lrcombine_inputs[1]))
	local grad_right_inputs = self.rightencoder:backward({input[1], input[2], input[4]}, self.rightencoder_dropout:backward(self.right_encode, grad_lrcombine_inputs[2]))
	self.gradInput = {grad_left_inputs[1] + grad_right_inputs[1], grad_left_inputs[2] + grad_right_inputs[2]}
	return self.gradInput
end
