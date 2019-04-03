CUDA_VISIBLE_DEVICES=5 th feat_train_connectingmodule.lua -input_h5 data/coco_embedding_softmax_thres50.h5 \
				-input_feat data/coco_resnet152_feats.h5 \
				-input_json data/coco_embedding.h5.mappings.json \
				-batch_size 64 \
				-dropout 0.5 \
				-learning_rate 1e-4 \
				-nEpochs 20 \
				-checkpoint_path checkpoints \
				-id embeddingtopdown_thres50 \
				-save_checkpoint_every 14906
