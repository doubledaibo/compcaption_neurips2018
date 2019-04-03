require 'nn'
require 'nngraph'

local components = {}

function components.AttCrit(batch_size)
	local inputs = {}
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	local r = inputs[1]
	local I = inputs[2]
	r = nn.Sqrt()(r)
	local rt = nn.Transpose({2, 3})(r)
	local rrt = nn.MM(false, false)({r, rt})
	local loss = nn.CSubTable()({I, rrt})
	loss = nn.Square()(loss)
	loss = nn.Sum(3, 3)(loss)
	loss = nn.Sum(2, 2)(loss)
	loss = nn.Sum(1, 1, batch_size)(loss)
	return nn.gModule(inputs, {loss})
end

function components.AttPred(att_hid_size, att_feat_size, att_size, num_class, dropout)
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()()) --att_feats
	local att_feats = inputs[1]
	local p_att_feats = nn.View(-1, att_feat_size)(att_feats)
	p_att_feats = nn.Linear(att_feat_size, att_hid_size)(p_att_feats)
	p_att_feats = nn.Tanh()(p_att_feats)
	local raw_weights = nn.Linear(att_hid_size, num_class)(p_att_feats)
	raw_weights = nn.View(-1, att_size, num_class)(raw_weights)
	local t_raw_weights = nn.Transpose({2, 3})(raw_weights)
	t_raw_weights = nn.View(-1, att_size)(t_raw_weights)
	local weights = nn.SoftMax()(t_raw_weights)
	weights = nn.View(-1, num_class, att_size)(weights)
	local att_res = nn.MM(false, false)({weights, att_feats})
	att_res = nn.View(-1, att_feat_size)(att_res)
	local pred = nn.Linear(att_feat_size, att_hid_size)(att_res)
	pred = nn.ReLU()(pred)
	pred = nn.Dropout(dropout)(pred)
	pred = nn.Linear(att_hid_size, att_hid_size)(pred)
	pred = nn.ReLU()(pred)
	pred = nn.Dropout(dropout)(pred)
	pred = nn.Linear(att_hid_size, 1)(pred)
	pred = nn.View(-1, num_class)(pred)
	pred = nn.Sigmoid()(pred)
	return nn.gModule(inputs, {pred, weights})	
end

function components.Attention(att_hid_size, att_feat_size, att_size, rnn_size)
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()()) --h
	table.insert(inputs, nn.Identity()()) --att_feats
	table.insert(inputs, nn.Identity()()) --p_att_feats
	local h = inputs[1]
	local att_feats = inputs[2]
   	local p_att_feats = inputs[3]
	local att = nn.View(-1, att_size, att_hid_size)(p_att_feats) -- consider batch as additional dim
	local att_h = nn.Linear(rnn_size, att_hid_size)(h)
	local att_h = nn.Replicate(att_size, 2, 3)(nn.View(-1, 1, att_hid_size)(att_h)) -- consider batch as additional dim
	local dot = nn.CAddTable()({att, att_h})
	local dot = nn.Tanh()(dot)
	local dot = nn.View(-1, att_hid_size)(dot) -- consider batch as additional dim
	local dot = nn.Linear(att_hid_size, 1)(dot)
	local dot = nn.View(-1, att_size)(dot) -- consider batch as additional dim
	local weight = nn.SoftMax()(dot)
	local weight = nn.View(1, -1):setNumInputDims(1)(weight) -- consider batch as minibatch
	local att_feats = nn.View(-1, att_size, rnn_size)(att_feats) -- consider batch as additional dim
	local att_res = nn.MM(false, false)({weight, att_feats})
	local att_res = nn.View(rnn_size):setNumInputDims(2)(att_res) -- consider batch as minibatch
	return nn.gModule(inputs, {att_res})
end


function components.TreeLSTMCell(input_size, rnn_size, dropout)
	local inputs = {}
	table.insert(inputs, nn.Identity()()) 
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	local x
	local c_left = inputs[2]
	local c_right = inputs[3]
	x = nn.JoinTable(1, 1)(inputs[1])
	
	local i2h = nn.Linear(input_size, 5 * rnn_size)(x)

	local reshaped = nn.Reshape(5, rnn_size)(i2h)
	local n1, n2, n3, n4, n5 = nn.SplitTable(2)(reshaped):split(5)
	local in_gate = nn.Sigmoid()(n1)
	local forget_gate_left = nn.Sigmoid()(n2)
	local forget_gate_right = nn.Sigmoid()(n3)
	local out_gate = nn.Sigmoid()(n4)
	local in_transform = nn.Tanh()(n5)
	local next_c = nn.CAddTable()({
			nn.CMulTable()({forget_gate_left, c_left}),
			nn.CMulTable()({forget_gate_right, c_right}),
			nn.CMulTable()({in_gate, in_transform})
		})
	-- gated cells form the output
	local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
	next_c = nn.Dropout(dropout)(next_c)
	next_h = nn.Dropout(dropout)(next_h)
	local outputs = {}
	table.insert(outputs, next_h)
	table.insert(outputs, next_c)
	return nn.gModule(inputs, outputs)
end

function components.LSTMCell(input_size, rnn_size, multiple_inputs)
	multiple_inputs = multiple_inputs or false
	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
	table.insert(inputs, nn.Identity()()) -- prev_h
	table.insert(inputs, nn.Identity()()) -- prev_c

	local x
	if multiple_inputs then
		x = nn.JoinTable(1, 1)(inputs[1])
	else
		x = inputs[1]
	end
	local prev_h = inputs[2]
	local prev_c = inputs[3]
	
	local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
	local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
	local all_input_sums = nn.CAddTable()({i2h, h2h})

	local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
	local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
	local in_gate = nn.Sigmoid()(n1)
	local forget_gate = nn.Sigmoid()(n2)
	local out_gate = nn.Sigmoid()(n3)
	local in_transform = nn.Tanh()(n4)
	local next_c = nn.CAddTable()({
			nn.CMulTable()({forget_gate, prev_c}),
			nn.CMulTable()({in_gate, in_transform})
		})
	-- gated cells form the output
	local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
	local outputs = {}
	table.insert(outputs, next_h)
	table.insert(outputs, next_c)
	return nn.gModule(inputs, outputs)
end

function components.RNNCell(img_size, rnn_size, rawh_size)
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()()) --img
	table.insert(inputs, nn.Identity()()) --lh
	table.insert(inputs, nn.Identity()()) --rh
	local img = inputs[1]
	local lh = inputs[2]
	local rh = inputs[3]
	local img2h = nn.Linear(img_size, rawh_size)(img)
	local lh2h = nn.Linear(rnn_size, rawh_size)(lh)
	local rh2h = nn.Linear(rnn_size, rawh_size)(rh)
	local raw_h = nn.Tanh()(nn.CAddTable()({img2h, lh2h, rh2h}))
	table.insert(outputs, raw_h)
	return nn.gModule(inputs, outputs)
end

function components.HCell(word_size, rawh_size, rnn_size)
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()()) --word
	table.insert(inputs, nn.Identity()()) --raw_h
	local x = inputs[1]
	local raw_h = inputs[2]
	local x2h = nn.Linear(word_size, rnn_size)(x)
	local h2h = nn.Linear(rawh_size, rnn_size)(raw_h)
	local h_embed = nn.Tanh()(nn.CAddTable()({x2h, h2h}))
	table.insert(outputs, h_embed)
	return nn.gModule(inputs, outputs)
end

return components
