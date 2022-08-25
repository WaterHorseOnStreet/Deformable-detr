set -x

export CUDA_VISIBLE_DEVICES=0
#export TORCH_DISTRIBUTED_DETAIL=DEBUG
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file cityperson --cityperson_path ../data/ --batch_size 1 \
--enc_layers 6 --dec_layers 6 --dim_feedforward 1024 --num_feature_levels 1 --model_name deformable_detr --output_dir ./output/random_train_true_test \
#--load_state False
# python main.py --dataset_file cityperson --cityperson_path ../../../dataset/cityscapes/ --batch_size 1 \
# --enc_layers 3 --dec_layers 3 --output_dir ./output 
