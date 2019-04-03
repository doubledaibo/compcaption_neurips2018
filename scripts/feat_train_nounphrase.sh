CUDA_VISIBLE_DEVICES=1 th feat_train_nounphrase.lua -input_h5 data/coco_nounphrase_multilabel_len5_thres50.h5 \
				-input_feat data/coco_resnet152_feats.h5 \
				-input_json data/coco_nounphrase_multilabel_len5_thres50.h5.mappings.json \
				-batch_size 256 \
				-nEpochs 50 \
				-checkpoint_path checkpoints \
				-id nounphrase_multilabel_len5_thres50 \
				-learning_rate 1e-4 \
				-save_checkpoint_every 2210
