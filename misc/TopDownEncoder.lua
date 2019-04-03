require 'nn'
require 'misc.LookupTableMaskZero'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local components = require 'misc.components'

local layer, parent = torch.class('nn.TopDownEncoder', 'nn.Module')
function layer:__init(cfg)
	parent.__init(self)

	self.input_vocab_size = cfg.input_vocab_size
	self.input_encoding_size = cfg.input_encoding_size
	self.rnn_size = cfg.rnn_size
	self.dropout = cfg.dropout
	self.seq_length = cfg.seq_length
	self.fc_feat_size = cfg.fc_feat_size 
	self.att_hid_size = cfg.att_hid_size
	self.att_size = cfg.att_size
	self.att_lstm = components.LSTMCell(self.input_encoding_size + cfg.rnn_size * 2, self.rnn_size, true)
	self.lang_lstm = components.LSTMCell(cfg.rnn_size * 2, self.rnn_size, true)
	self.attention = components.Attention(cfg.att_hid_size, cfg.fc_feat_size, cfg.att_size, cfg.rnn_size)
	self.att_embed = nn.Sequential()
					:add(nn.View(-1, self.fc_feat_size))
					:add(nn.Linear(self.fc_feat_size, self.rnn_size))
					:add(nn.ReLU())
					:add(nn.Dropout(cfg.dropout))
	self.w_embed = nn.Sequential()
				:add(nn.LookupTableMaskZero(self.input_vocab_size + 1, self.input_encoding_size))
	self.img_embed = nn.Sequential()
				:add(nn.Linear(self.fc_feat_size, self.input_encoding_size))
				:add(nn.ReLU())
				:add(nn.Dropout(self.dropout))	
	self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
	self.start_token = torch.Tensor()
	self.grad_img_feats = torch.Tensor()
	self.grad_att_feats = torch.Tensor()
	self.grad_p_att_feats = torch.Tensor()
	self:_createInitState(1) 
end

function layer:_createInitState(batch_size)
	assert(batch_size ~= nil, 'batch size must be provided')
	if not self.init_state then self.init_state = {} end -- lazy init
	for h = 1, 2 do
		if self.init_state[h] then
			if self.init_state[h]:size(1) ~= batch_size then
				self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
			end
		else
			self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
		end
	end
	self.num_state = #self.init_state
end

function layer:shareThinClone(thin_copy)
	thin_copy.att_lstm:share(self.att_lstm, 'weight', 'bias')
	thin_copy.lang_lstm:share(self.lang_lstm, 'weight', 'bias')
	thin_copy.attention:share(self.attention, 'weight', 'bias')
	thin_copy.att_embed:share(self.att_embed, 'weight', 'bias')
	thin_copy.ctx2att:share(self.ctx2att, 'weight', 'bias')
	thin_copy.w_embed:share(self.w_embed, 'weight', 'bias')
	thin_copy.img_embed:share(self.img_embed, 'weight', 'bias')
end

function layer:createClones()
	print('constructing clones inside the FCModel')
	self.att_lstms = {self.att_lstm}
	self.lang_lstms = {self.lang_lstm}
	self.attentions = {self.attention}
	self.w_embeds = {self.w_embed}
	for t = 2, self.seq_length + 1 do
		self.att_lstms[t] = self.att_lstm:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lang_lstms[t] = self.lang_lstm:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.attentions[t] = self.attention:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.w_embeds[t] = self.w_embed:clone('weight', 'bias', 'gradWeight', 'gradBias')
	end
end

function layer:getModulesList()
	return {self.att_lstm, self.lang_lstm, self.attention, self.w_embed, self.img_embed, self.att_embed, self.ctx2att}
end

function layer:parameters()
	local params = {}
	local grad_params = {}
	local modules = self:getModulesList()
	for dummy, module in pairs(modules) do
		local p, g = module:parameters()
		for k, v in pairs(p) do table.insert(params, v) end
		for k, v in pairs(g) do table.insert(grad_params, v) end
	end
	
	return params, grad_params
end

function layer:training()
	for k, v in pairs(self.lang_lstms) do v:training() end
	for k, v in pairs(self.att_lstms) do v:training() end
	for k, v in pairs(self.attentions) do v:training() end
	for k, v in pairs(self.w_embeds) do v:training() end
	self.img_embed:training()
end

function layer:evaluate()
	for k, v in pairs(self.lang_lstms) do v:evaluate() end
	for k, v in pairs(self.att_lstms) do v:evaluate() end
	for k, v in pairs(self.attentions) do v:evaluate() end
	for k, v in pairs(self.w_embeds) do v:evaluate() end
	self.img_embed:evaluate()
end

function layer:updateOutput(input)
	local seq = input[3]
	if self.w_embeds == nil then self:createClones() end 

	assert(seq:size(1) == self.seq_length)
	local batch_size = input[1]:size(1)

	self.output:resize(batch_size, self.rnn_size):zero()
	self.start_token:resize(batch_size):fill(self.input_vocab_size + 1)
	self:_createInitState(batch_size)
	self.mask = 1 - torch.eq(seq, 0)
	self.mask = torch.repeatTensor(self.mask, self.rnn_size, 1, 1):permute(2, 3, 1)
--	self.mask = self.mask:expand(self.seq_length, batch_size, self.rnn_size)
	self.mask = self.mask:cuda()
	self.img_feats = self.img_embed:forward(input[1])
	self.att_feats = self.att_embed:forward(input[2])
	self.p_att_feats = self.ctx2att:forward(self.att_feats)
	
	self.att_states = {[0] = self.init_state}
	self.lang_states = {[0] = self.init_state}
	self.att_inputs = {}
	self.lang_inputs = {}
	self.attention_inputs = {}
	self.embed_inputs = {}
	self.tmax = 0

	for t = 1, self.seq_length + 1 do
		local xt
		local it
		if t == 1 then
			it = self.start_token
		else
			it = seq[t - 1]:clone()
		end
		if torch.sum(it) == 0 then
			break 
		end
		self.embed_inputs[t] = it
		xt = self.w_embeds[t]:forward(it)
		self.tmax = t
	
		self.att_inputs[t] = {{self.lang_states[t - 1][1], self.img_feats, xt}, unpack(self.att_states[t - 1])}
		local att_out = self.att_lstms[t]:forward(self.att_inputs[t])
		self.att_states[t] = {}
		for i = 1, self.num_state do table.insert(self.att_states[t], att_out[i]) end
		self.attention_inputs[t] = {att_out[1], self.att_feats, self.p_att_feats}
		local att = self.attentions[t]:forward(self.attention_inputs[t])
		self.lang_inputs[t] = {{att, att_out[1]}, unpack(self.lang_states[t - 1])}
		local lang_out = self.lang_lstms[t]:forward(self.lang_inputs[t])	
		self.lang_states[t] = {}
		for i = 1, self.num_state do table.insert(self.lang_states[t], lang_out[i]) end
		if t == 1 then
			self.output:add(self.lang_states[t][1])
		else
			self.output:cmul(1 - self.mask[t - 1])
			self.output:add(torch.cmul(self.lang_states[t][1], self.mask[t - 1]))
		end
	end
--	for b = 1, batch_size do
--		for t = self.tmax, 2, -1 do
--			if seq[t - 1][b] ~= 0 then
--				assert(torch.sum(1 - torch.eq(self.output[b], self.lstm_states[t][1][b])) == 0)
--				break
--			end
--		end
--	end
	return self.output
end

function layer:updateGradInput(input, gradOutput)
	local grad_lang_state = self.init_state
	local grad_att_state = self.init_state
	self.grad_att_feats:resizeAs(self.att_feats):zero()
	self.grad_p_att_feats:resizeAs(self.p_att_feats):zero()
	self.grad_img_feats:resizeAs(self.img_feats):zero()
	for t=self.tmax, 1, -1 do
		local grad_lang_inputs
		if t == 1 then
			grad_lang_inputs = self.lang_lstms[t]:backward(self.lang_inputs[t], {grad_lang_state[1], grad_lang_state[2]})
		else
			grad_lang_inputs = self.lang_lstms[t]:backward(self.lang_inputs[t], {torch.cmul(gradOutput, self.mask[t - 1]) + grad_lang_state[1], grad_lang_state[2]})
			gradOutput:cmul(1 - self.mask[t - 1])
		end
		grad_lang_state = {}
		for k = 2, self.num_state + 1 do table.insert(grad_lang_state, grad_lang_inputs[k]) end
		local grad_attention_inputs = self.attentions[t]:backward(self.attention_inputs[t], grad_lang_inputs[1][1])
		self.grad_att_feats:add(grad_attention_inputs[2])
		self.grad_p_att_feats:add(grad_attention_inputs[3])
		local grad_att_inputs = self.att_lstms[t]:backward(self.att_inputs[t], {grad_att_state[1] + grad_attention_inputs[1] + grad_lang_inputs[1][2], grad_att_state[2]})
		grad_att_state = {}
		for k = 2, self.num_state + 1 do table.insert(grad_att_state, grad_att_inputs[k]) end
		grad_lang_state[1]:add(grad_att_inputs[1][1])
		self.grad_img_feats:add(grad_att_inputs[1][2])
		self.w_embeds[t]:backward(self.embed_inputs[t], grad_att_inputs[1][3])
	end
	self.grad_att_feats:add(self.ctx2att:backward(self.att_feats, self.grad_p_att_feats))
	local grad_att_feats = self.att_embed:backward(input[2], self.grad_att_feats)
	local grad_img_feats = self.img_embed:backward(input[1], self.grad_img_feats)
	self.gradInput = {grad_img_feats, grad_att_feats}
	return self.gradInput
end


