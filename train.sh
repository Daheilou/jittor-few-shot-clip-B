CUDA_VISIBLE_DEVICES=0
train_data='./data/train.txt'
val_data='./data/val.txt'
seen_class='../Dataset/classes.txt'
img_dir='../Dataset'
pretrain_vit_model='./pretrain/ViT-B-32.pkl'
pretrain_rn_model='./pretrain/RN101.pkl'
pretrain_convnext_model='./pretrain/convnextv2_base_1k_224_ema.pkl'
save_lp_vit_model='./out/LP-ViT.pkl'
save_lp_vit_vecer='./out/LP-ViT-vecer.pkl'
save_lp_rn_model='./out/LP-RN.pkl'
save_lp_rn_vecer='./out/LP-RN-vecer.pkl'
save_convnext_model='./out/convnextv2-base.pkl'
batch_size=32


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python few_lp.py \
    --seed=2480 \
    --img_dir=$img_dir \
    --optimizer_lr_steps 1000 \
    --train_data=$train_data \
    --val_data=$val_data \
    --pretrain_model=$pretrain_vit_model \
    --seen_class=$seen_class \
    --save_lp_model=$save_lp_vit_model \
    --save_lp_vecer=$save_lp_vit_vecer
    

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python few_lp.py \
--seed=960 \
--img_dir=$img_dir \
--optimizer_lr_steps 500 \
--train_data=$train_data \
--val_data=$val_data \
--pretrain_model=$pretrain_rn_model \
--seen_class=$seen_class \
--save_lp_model=$save_lp_rn_model \
--save_lp_vecer=$save_lp_rn_vecer

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_dog.py \
--seed=42 \
--imgs_dir=$img_dir \
--train_data=$train_data \
--val_data=$val_data \
--pretrain_convnext_model=$pretrain_convnext_model \
--save_model=$save_convnext_model \
--batch_size=$batch_size