CUDA_VISIBLE_DEVICES=0
test_img='../TestSetB/'
all_class='../Dataset/classes_b.txt'
pretrain_vit_model='./pretrain/ViT-B-32.pkl'
pretrain_rn_model='./pretrain/RN101.pkl'
pretrain_convnext_model='./out/convnextv2-base.pkl'
vit_alpha_vec='./out/LP-ViT-vecer.pkl'
vit_LP='./out/LP-ViT.pkl'
rn_alpha_vec='./out/LP-RN-vecer.pkl'
rn_LP='./out/LP-RN.pkl'
out='result.txt'



CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python test.py \
--test_img=$test_img \
--all_class=$all_class \
--pretrain_vit_model=$pretrain_vit_model \
--pretrain_rn_model=$pretrain_rn_model \
--pretrain_convnext_model=$pretrain_convnext_model \
--vit_alpha_vec=$vit_alpha_vec \
--vit_LP=$vit_LP \
--rn_alpha_vec=$rn_alpha_vec \
--rn_LP=$rn_LP \
--output=$out