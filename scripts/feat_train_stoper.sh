CUDA_VISIBLE_DEVICES=3 th feat_train_stoper.lua -input_h5 data/coco_stops.h5 \
				-input_feat data/coco_resnet152_feats.h5 \
				-input_json data/coco_embedding.h5.mappings.json \
				-batch_size 256 \
				-checkpoint_path checkpoints \
				-id stoper \
				-save_checkpoint_every 19544
