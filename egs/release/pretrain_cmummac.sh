current_time=$(date "+%Y.%m.%d-%H.%M.%S")
dataset=cmummac
dataset_base_path=/large/mfzhang/data/data_release/cmu-mmac-release
data_train=${dataset_base_path}/cav_label/train_pretrain/labels_6785.json
data_val=${dataset_base_path}/cav_label/test_pretrain/labels_2808.json
label_csv=${dataset_base_path}/cav_label/class_labels_indices.csv
pretrain_path=${dataset_base_path}/../videomae_adapt_ckpt/video_imu_beforepretrain_model_small_only_encoder.pth
exp_dir=${dataset_base_path}/evi-mae-exp/${current_time}-pretrain-${dataset}


model=evi-mae
batch_size=24
epoch=300
lrscheduler_start=100
lrscheduler_decay=0.5
lrscheduler_step=100
load_prepretrain=True
contrast_loss_weight=0.01
mae_loss_weight=1.0
norm_pix_loss=True # if True, no inpainting
pretrain_modality=both # imu, video, or both

imu_plot_type=stft
imu_channel_num=12 # wear 12
imu_target_length=320 # need to be 16x, fbank 151->144 for wear; 51->48 for cmu
imu_plot_height=128 # need to be 16x # in dataloader.py 128 is set do not change it
imu_patch_size=16 
imu_dataset_mean=-69.8
imu_dataset_std=23.5
imu_masking_ratio=0.75
imu_mask_mode=unstructured # or time, or freq, or tf

imu_enable_graph=True
imu_graph_net=gin
imu_graph_masking_ratio=0.5

# small
imu_encoder_embed_dim=768
imu_encoder_depth=11
imu_encoder_num_heads=12

video_img_size=224
video_patch_size=16
video_encoder_num_classes=0
video_decoder_num_classes=1536
video_mlp_ratio=4
video_qkv_bias=True
video_masking_ratio=0.9

video_encoder_embed_dim=384
video_encoder_depth=12
video_encoder_num_heads=6 
video_decoder_embed_dim=192 
video_decoder_num_heads=3

image_as_video=False

mkdir -p $exp_dir
echo $exp_dir

CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_evimae_pretrain.py --model ${model} --dataset ${dataset} \
--data-train ${data_train} --data-val ${data_val} --exp-dir $exp_dir \
--label-csv ${label_csv} \
--n-epochs ${epoch} --batch-size $batch_size --save_model True \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--warmup True \
--norm_pix_loss ${norm_pix_loss} \
--pretrain_path ${pretrain_path} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--imu_plot_type ${imu_plot_type} --imu_plot_height ${imu_plot_height} --imu_patch_size ${imu_patch_size} \
--imu_dataset_mean ${imu_dataset_mean} --imu_dataset_std ${imu_dataset_std} --imu_channel_num ${imu_channel_num} \
--imu_masking_ratio ${imu_masking_ratio} --imu_mask_mode ${imu_mask_mode} --imu_target_length ${imu_target_length} \
--video_img_size ${video_img_size} --video_patch_size ${video_patch_size} --video_encoder_num_classes ${video_encoder_num_classes} \
--video_decoder_num_classes ${video_decoder_num_classes} --video_mlp_ratio ${video_mlp_ratio} --video_qkv_bias ${video_qkv_bias} \
--video_encoder_embed_dim ${video_encoder_embed_dim} --video_encoder_depth ${video_encoder_depth} \
--video_encoder_num_heads ${video_encoder_num_heads} --video_decoder_embed_dim ${video_decoder_embed_dim} \
--video_decoder_num_heads ${video_decoder_num_heads} --video_masking_ratio ${video_masking_ratio} \
--imu_encoder_embed_dim ${imu_encoder_embed_dim} --imu_encoder_depth ${imu_encoder_depth} \
--imu_encoder_num_heads ${imu_encoder_num_heads} --load_prepretrain ${load_prepretrain} \
--imu_enable_graph ${imu_enable_graph} --imu_graph_net ${imu_graph_net} \
--pretrain_modality ${pretrain_modality} --image_as_video ${image_as_video} --imu_graph_masking_ratio ${imu_graph_masking_ratio} 