require 'torch'
require 'nn'
require 'nngraph'
require 'misc.FeatDataLoaderResNetEval'

local utils = require 'misc.utils'
require 'misc.TopDownConnectingModule'
require 'misc.TopDownStoper'

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
cmd:option('-nounphrase_encode_h5', '', '')   -- encoded nounphrase, obtained from encode_nounphrase.py
cmd:option('-vocab_mapping', '', '')  -- vocab mappings
cmd:option('-simword_mapping', '', '')  -- obtained using wordnet and thesaurus
cmd:option('-combination_order', '', '') -- gt combination orders
cmd:option('-nounphrase_h5', '', '')  -- gt for nounphrase prediction
cmd:option('-nounphrasepred_h5', '', '') -- pred for nounphrase prediction, obtained from feat_eval_nounphrase.lua
cmd:option('-combination_h5', '', '') -- hdf for the connecting module  
cmd:option('-stoper', '', '') -- evaluation module
cmd:option('-lm', '', '') -- connecting module
cmd:option('-dataset', '', '') 
cmd:option('-nounphrase_usegt', 0, '')
cmd:option('-combination_usegt', 0, '') 
cmd:option('-split', 'val', '')
cmd:option('-num_nounphrases', 20, '')
cmd:option('-thres_nounphrase', 0.2, '')
cmd:option('-thres_embedding', 0.001, '')
cmd:option('-thres_stop', 0.7, '')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-id', "1", '')
cmd:option('-beam_size', 3, '') -- beam_size over composing pairs
cmd:option('-mid_beam_size', 3, '') -- beam_size over connecting phrases
cmd:option('-num_sample', -1, '')
cmd:option('-val_idx_json', '', '')
cmd:option('-verbose', 1, '')
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

local loader = DataLoader(cfg)

---evaluation module
if cfg.stoper ~= '' then
	print("load stoper from " .. cfg.stoper)
	local loaded_checkpoint = torch.load(cfg.stoper)
	protos.stoper = loaded_checkpoint.stoper:cuda()
else
	error("need a stoper")
end

---connecting module
if cfg.lm ~= '' then
	print("load lm from " .. cfg.lm)
	local loaded_checkpoint = torch.load(cfg.lm)
	protos.lm = loaded_checkpoint.protos.lm:cuda()
else
	error("need a lm")
end

protos.stoper:createClones()
protos.lm:createClones()

protos.stoper:evaluate()
protos.lm:evaluate()

protos.rnn_size = protos.lm.leftencoder.rnn_size

protos.simword_mapping = utils.read_json(cfg.simword_mapping)
protos.input_itow = loader:getInputItoW()

function avoidRepeat(subphrases, subphrase)
	local word1 = protos.input_itow[tostring(subphrase.last_token)]
	for k, v in pairs(subphrases) do
		if subphrase.last_token == v.last_token then
			return false
		end
		local word2 = protos.input_itow[tostring(v.last_token)]
		if protos.simword_mapping[word1 .. "-" .. word2] == 1 or protos.simword_mapping[word2 .. "-" .. word1] == 1 then
			return false
		end
	end
	for k, v in pairs(subphrases) do
		local dist = torch.sum(torch.pow(v.embed_vector - subphrase.embed_vector, 2))
		if dist < cfg.thres_embedding then
			return false
		end
	end
	return true
end

function getNounPhrases(pred)
	local top_prob, top_idx = pred:topk(math.min(pred:size(2), cfg.num_nounphrases), 2, true, true)
	local subphrases = {}		
	local len = loader.combination_length
	for i = 1, top_idx:size(2) do
		if top_prob[1][i] < cfg.thres_nounphrase and #subphrases > 0 then break end
		local cur_nounphrase = loader:getNounPhraseEncode(top_idx[1][i]):cuda()
		local nounphrase = torch.LongTensor(len, 1):zero():cuda()
		local l = cur_nounphrase:size(1)
		local last_token = -1
		for j = 1, l do
			nounphrase[j][1] = cur_nounphrase[j][1]
			if cur_nounphrase[j][1] ~= 0 then
				last_token = cur_nounphrase[j][1]
			end
		end			
		local subphrase = {}
		subphrase.phrase = nounphrase
		subphrase.lp = top_prob[1][i]
		subphrase.last_token = last_token
		subphrase.embed_vector = torch.FloatTensor(1, protos.rnn_size * 2):zero()
		local left = protos.lm.leftencoder:forward({protos.img_feat, protos.conv_feat, nounphrase}):float()
		local right = protos.lm.rightencoder:forward({protos.img_feat, protos.conv_feat, nounphrase}):float()
--		left = torch.renorm(left, 1, 1, 1)
--		right = torch.renorm(right, 1, 1, 1)
		subphrase.embed_vector[{{1, 1}, {1, protos.rnn_size}}] = left
		subphrase.embed_vector[{{1, 1}, {protos.rnn_size + 1, protos.rnn_size * 2}}] = right
		subphrase.embed_vector = torch.renorm(subphrase.embed_vector, 1, 1, 1)
		if avoidRepeat(subphrases, subphrase) then
			table.insert(subphrases, subphrase)
		end
	end
	return subphrases
end 

function createOption(left, right)
	local option = {}
	option.l = left
	option.r = right
	return option
end

function updateSubphrases(new_phrases, new_srcs, options, perm)
	local new_processes = {}
	local m = #protos.processes
	for i = 1, #new_phrases do
		local k = math.floor(new_srcs[i] / (m * cfg.mid_beam_size)) + 1
		local j = new_srcs[i] % (m * cfg.mid_beam_size)
		if j == 0 then
			j = m * cfg.mid_beam_size
			k = k - 1
		end
		if j % cfg.mid_beam_size == 0 then
			j = j / cfg.mid_beam_size
		else
			j = math.floor(j / cfg.mid_beam_size) + 1
		end
		local option = options[perm[k]]
		local new_process = {}
		table.insert(new_process, new_phrases[i])
		for t = 1, #protos.processes[j] do
			if t ~= option.l and t ~= option.r then	
				table.insert(new_process, protos.processes[j][t])
			end
		end
		table.insert(new_processes, new_process)
	end
	protos.processes = new_processes 
end

function getOptionAccordingtoidf(idf)
	local n = #protos.subphrases
	for i = 1, n do
		if "(" .. protos.subphrases[i].idf .. "+)" == idf then
			return i, 0
		elseif "(+" .. protos.subphrases[i].idf .. ")" == idf then
			return 0, i 
		end 
	end
	for i = 1, n do
		for j = 1, n do
			if i ~= j and "(" .. protos.subphrases[i].idf .. "+" .. protos.subphrases[j].idf .. ")" == idf then
				return i, j
			end	
		end
	end
	print("could not find the correct combining pair")
	aaaa:clone()
end 

function getMSeq(left, right, lp)
	local b = left:size(2)
	local pred = protos.lm:forward({torch.repeatTensor(protos.img_feat, b, 1), torch.repeatTensor(protos.conv_feat, b, 1, 1), left, right})
	local mseq = torch.LongTensor(loader.combination_length, b * cfg.mid_beam_size):zero():cuda()
	local total_lp = torch.FloatTensor(b * cfg.mid_beam_size):zero()
	for k = 1, b do
		local s, top_idx = pred[k]:topk(cfg.mid_beam_size, 1, true, true)
		for j = 1, cfg.mid_beam_size do	
			if top_idx[j] == loader.num_mseq + 1 then
				total_lp[(k - 1) * cfg.mid_beam_size + j] = -1000  --invalid pair
			else
				total_lp[(k - 1) * cfg.mid_beam_size + j] = torch.exp(s[j]) + lp[k]
				mseq[{{}, {(k - 1) * cfg.mid_beam_size + j, (k - 1) * cfg.mid_beam_size + j}}] = loader:getMSeq(top_idx[j], top_idx[j])
			end
		end
	end
	return mseq, total_lp
end

function getCapUseOrder()
--	local input_wtoi = loader:getInputWtoI()
--	local output_itow = loader:getOutputItoW()

	local n = #protos.orders
	local len = loader.combination_length
	for t = 1, n do
		local l, r = getOptionAccordingtoidf(protos.orders[t])
		local left = torch.LongTensor(len, 1):zero():cuda()
		local right = torch.LongTensor(len, 1):zero():cuda()
		local lp = torch.FloatTensor(1):zero():cuda()
		if l ~= 0 then
			left[{{}, {1, 1}}] = protos.subphrases[l].phrase
			lp[1] = lp[1] + protos.subphrases[l].lp
		end
		if r ~= 0 then
			right[{{}, {1, 1}}] = protos.subphrases[r].phrase
			lp[1] = lp[1] + protos.subphrases[r].lp
		end		
		local mid, total_lp = getMSeq(left, right, lp)
		local new_phrase = torch.LongTensor(len, 1):zero():cuda()
		local base = 0
		for i = 1, len do
			if left[i][1] ~= 0 then
				new_phrase[base + i][1] = left[i][1]
			else
				base = i - 1
				break
			end
		end
		for i = 1, len do
			if mid[i][1] ~= 0 and mid[i][1] ~= loader.input_vocab_size + 1 and base + i <= len then
				new_phrase[base + i][1] = mid[i][1]
			else
				base = base + i - 1
				break
			end
		end
		for i = 1, len do
			if right[i][1] ~= 0 and base + i <= len then
				new_phrase[base + i][1] = right[i][1]
			else
				break
			end
		end
		local new_subphrase = {}
		new_subphrase.lp = total_lp[1]
		new_subphrase.phrase = new_phrase
		new_subphrase.idf = protos.orders[t]
		local new_table = {new_subphrase}
		for i = 1, #protos.subphrases do
			if i ~= l and i ~= r then
				table.insert(new_table, protos.subphrases[i])
			end
		end
		protos.subphrases = new_table
		collectgarbage()
	end
	return protos.subphrases[1].phrase, protos.subphrases[1].lp 
end

function getCap()
--	local input_wtoi = loader:getInputWtoI()
--	local output_itow = loader:getOutputItoW()
	protos.processes = {}
	table.insert(protos.processes, protos.subphrases)
	local done_phrases = {}
	while 1 do
		if #protos.processes == 0 then
			break
		end
		if #protos.processes[1] == 1 then
			if #done_phrases == 0 then
				for i = 1, #protos.processes do
					table.insert(done_phrases, protos.processes[i][1])
				end
			end
			break
		end
		local n = #protos.processes[1]
		local m = #protos.processes
		local options = {}
		for i = 1, n do 
			for j = 1, n do
				if i ~= j then table.insert(options, createOption(i, j)) end
			end
		end
		local perm = torch.range(1, #options)
	--	local perm = torch.randperm(#options)
		local len = loader.combination_length
	--	local b = math.min(#options, math.max(40, math.floor(#options * 0.2)))
		local b = #options
		local left = torch.LongTensor(len, b * m):zero():cuda()
		local right = torch.LongTensor(len, b * m):zero():cuda()
		local lp = torch.FloatTensor(b * m):zero()
		for i = 1, b do
			for k = 1, m do
				local idx = (i - 1) * m + k
				local l, r = options[perm[i]].l, options[perm[i]].r
				if l ~= 0 then
					left[{{}, {idx, idx}}] = protos.processes[k][l].phrase
					lp[idx] = lp[idx] + protos.processes[k][l].lp
				end
				if r ~= 0 then
					right[{{}, {idx, idx}}] = protos.processes[k][r].phrase
					lp[idx] = lp[idx] + protos.processes[k][r].lp
				end		
			end
		end
		local mid, total_lp = getMSeq(left, right, lp)
		local beam_size = math.min(cfg.beam_size, b * m * cfg.mid_beam_size)
		local toplp, topidx = total_lp:topk(beam_size, 1, true, true)
		local new_phrases = {}
		local new_srcs = {}
		for k = 1, beam_size do	
			local base = 0
			local idx = topidx[k]
			local idx2
			if idx % cfg.mid_beam_size == 0 then
				idx2 = idx / cfg.mid_beam_size
			else
				idx2 = math.floor(idx / cfg.mid_beam_size) + 1
			end
			local new_phrase = torch.LongTensor(len, 1):zero():cuda()
			for i = 1, len do
				if left[i][idx2] ~= 0 then
					new_phrase[base + i][1] = left[i][idx2]
				else
					base = i - 1
					break
				end
			end
			for i = 1, len do
				if mid[i][idx] ~= 0 and mid[i][idx] ~= loader.input_vocab_size + 1 and base + i <= len then
					new_phrase[base + i][1] = mid[i][idx]
				else
					base = base + i - 1
					break
				end
			end
			for i = 1, len do
				if right[i][idx2] ~= 0 and base + i <= len then
					new_phrase[base + i][1] = right[i][idx2]
				else
					break
				end
			end
			local stop_pred = protos.stoper:forward({protos.img_feat, protos.conv_feat, new_phrase})
			local new_subphrase = {}
			new_subphrase.lp = toplp[k]
			new_subphrase.phrase = new_phrase
			new_subphrase.stop = stop_pred[1][1]
			if stop_pred[1][1] >= cfg.thres_stop then
				table.insert(done_phrases, new_subphrase)
			else	
				table.insert(new_srcs, idx)
				table.insert(new_phrases, new_subphrase)
			end	
		end
		updateSubphrases(new_phrases, new_srcs, options, perm)
		collectgarbage()
	end
	local function compare(a, b) return a.lp > b.lp end
	table.sort(done_phrases, compare)
	return done_phrases[1].phrase, done_phrases[1].lp, done_phrases[1].stop
end

local nbatch = loader:getnBatch(cfg.split)
if cfg.num_sample ~= -1 then
	nbatch = cfg.num_sample
end
local input_itow = loader:getInputItoW()

loader:init_rand(cfg.split)
loader:reset_iterator(cfg.split)
local uses = {}
for i = 1, nbatch do
	uses[i] = 1
end
if cfg.val_idx_json ~= '' then
	local val_idx_json = utils.read_json(cfg.val_idx_json)
	uses = {}
	for k, v in pairs(val_idx_json) do
		uses[v] = 1
	end
end
local imgid_cell = {}
local predictions = {}
local cnt = 0
for n = 1, nbatch do
	protos.img_feat, protos.conv_feat, protos.img_id = loader:nextImgConv(cfg.split)
	if uses[n] ~= nil then
		cnt = cnt + 1
		if cnt % 5 == 0 then
			print(cnt .. " / " .. nbatch)
		end
		protos.img_feat = protos.img_feat:cuda()
		protos.conv_feat = protos.conv_feat:cuda()
		local pred
		if cfg.combination_usegt == 0 then
			if cfg.nounphrase_usegt == 1 then
				pred = loader:getGTNounPhrasePreds(protos.img_id)
				protos.subphrases = loader:getGTNounPhrases(protos.img_id)
			else
				pred = loader:getNounPhrasePreds(protos.img_id)
		--	end
				protos.subphrases = getNounPhrases(pred)
			end
		end
		local seq, seq_lp
	--	local best_seq, best_seq_lp
	--	best_seq = nil
		if cfg.combination_usegt == 1 then
			protos.subphrases, protos.orders = loader:getGTOrders(protos.img_id)
			seq, seq_lp = getCapUseOrder()
		else
			seq, seq_lp, score = getCap()
		end	
		local sent = net_utils.decode_sequence(input_itow, seq)
		if imgid_cell[protos.img_id] == nil then
			imgid_cell[protos.img_id] = 1
			entry = {image_id = protos.img_id, caption = sent[1]}
			table.insert(predictions, entry)
			if cnt < 50 and cfg.verbose == 1 then
				print(string.format('image %s: %s', entry.image_id, entry.caption))
			end
		end
	end 
end

net_utils.language_eval(predictions, cfg)
utils.write_json("evaloutputs/" .. cfg.id .. "_out.json", predictions)
