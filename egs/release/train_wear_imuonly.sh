current_time=$(date "+%Y.%m.%d-%H.%M.%S")
dataset=wear
dataset_base_path=/large/mfzhang/data/data_release/wear-release
data_train=${dataset_base_path}/cav_label/train_finetune/labels_2553.json
data_val=${dataset_base_path}/cav_label/test_finetune/labels_1136.json
label_csv=${dataset_base_path}/cav_label/class_labels_indices.csv
pretrain_path=None
exp_dir=${dataset_base_path}/evi-mae-exp/${current_time}-finetune-${dataset}

model=evi-mae-ft
ftmode=ft_imuonly
bal=None
lr=5e-5
batch_size=16
epoch=200
lrscheduler_start=60
lrscheduler_decay=0.5
lrscheduler_step=60
wa=True
wa_start=40
wa_end=200
wa_num=12
freeze_base=False
head_lr=100
lr_adapt=False
base_lr=1

n_class=18
label_smooth=0.1
noise=True
freqm=48
timem=192
mixup=0

imu_plot_type=stft
imu_channel_num=12
imu_target_length=320
imu_plot_height=128
imu_patch_size=16 
imu_dataset_mean=-51.17
imu_dataset_std=26.19

imu_enable_graph=True
imu_graph_net=gin
imu_two_stream=False

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

metrics=acc

mkdir -p $exp_dir
echo $exp_dir

CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_evimae_ft.py --model ${model} --dataset ${dataset} \
--data-train ${data_train} --data-val ${data_val} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} --noise ${noise} \
--loss BCE --metrics mAP --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --wa_num ${wa_num} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--imu_plot_type ${imu_plot_type} --imu_plot_height ${imu_plot_height} --imu_patch_size ${imu_patch_size} \
--imu_dataset_mean ${imu_dataset_mean} --imu_dataset_std ${imu_dataset_std} --imu_channel_num ${imu_channel_num} \
--imu_target_length ${imu_target_length} \
--video_img_size ${video_img_size} --video_patch_size ${video_patch_size} --video_encoder_num_classes ${video_encoder_num_classes} \
--video_decoder_num_classes ${video_decoder_num_classes} --video_mlp_ratio ${video_mlp_ratio} --video_qkv_bias ${video_qkv_bias} \
--video_encoder_embed_dim ${video_encoder_embed_dim} --video_encoder_depth ${video_encoder_depth} \
--video_encoder_num_heads ${video_encoder_num_heads} --video_decoder_embed_dim ${video_decoder_embed_dim} \
--video_decoder_num_heads ${video_decoder_num_heads} --video_masking_ratio ${video_masking_ratio} \
--imu_encoder_embed_dim ${imu_encoder_embed_dim} --imu_encoder_depth ${imu_encoder_depth} \
--imu_encoder_num_heads ${imu_encoder_num_heads} --num-workers 32 \
--imu_enable_graph ${imu_enable_graph} --imu_graph_net ${imu_graph_net} --base_lr ${base_lr} \
--imu_two_stream ${imu_two_stream} --metrics ${metrics}