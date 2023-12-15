#!/bin/bash
CUDA_USED=6
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_USED}
now=$(date "+%Y-%m-%d-%Hh")
data_tag="tw_mm_s1_ocr"
#cur_model="mixture_img_text_multi_head_att_h4_d128"
cur_model="mixture_text_img_multi_head_att_h4_d128"
#ft_model="mixture_img_text_multi_head_att_h4_d256_combine_direct"
#seed=159
batch_size=64
#tag=$[$RANDOM%5000+100]
repreprocess_flag=0
use_itm=True
use_kl=True
use_type=True
is_debug=False

data_suffix='_type'
test_src='./data/{}/test_ocr_kw.txt '
#img_ext_model="complex_resnet"
img_ext_model="vgg"
preprocess_dir="../processed_data/${data_tag}"
data_dir="../data/${data_tag}"
raw_data_path="../data/{}"
res_fn="../sh/results/mixture_multi_head_ocr_pretrain_${seed}_${tag}_${now}.csv"

seed=159
tag=1111

echo "=================== p1 : Start training ========================"
cmd="CUDA_VISIBLE_DEVICES=${CUDA_USED} python ./train_itm.py \
        -cur_model ${cur_model} \
        -data_tag ${data_tag} \
        -batch_size ${batch_size} \
        -seed ${seed} \
        -data_suffix ${data_suffix} \
        -test_src ${test_src} \
        -res_fn ${res_fn} \
        -raw_data_path ${raw_data_path} \
        -tag ${tag} \
        -img_ext_model ${img_ext_model} \
        "
if [ ${use_itm} == True ]; then
    cmd+="-itm "
fi
if [ ${use_kl} == True ]; then
    cmd+="-use_kl "
fi
if [ ${use_type} == True ]; then
    cmd+="-use_type "
fi
if [ ${is_debug} == True ]; then
    cmd+="-debug 1 "
fi

echo $cmd
eval $cmd

phase2_cur_model="${cur_model}_combine_direct"
phase2_res_fn="../sh/results/mixture_multi_head_ocr_combine_direct_${seed}_${now}.csv"
model_path="../models/${cur_model}-${data_tag}-${seed}/${tag}kl${use_kl}itm${use_itm}best1.ckpt"

echo "=================== p2 : Start training  ========================"
cmd="CUDA_VISIBLE_DEVICES=${CUDA_USED} python ./train_itm.py \
        -cur_model ${phase2_cur_model} \
        -model_path ${model_path} \
        -data_tag ${data_tag} \
        -batch_size ${batch_size} \
        -seed ${seed} \
        -res_fn ${res_fn} \
        -data_suffix ${data_suffix} \
        -test_src ${test_src} \
        -raw_data_path ${raw_data_path} \
        -fix_classifier 1 \
        -tag ${tag} \
        -img_ext_model ${img_ext_model} \
        "
if [ ${use_itm} == True ]; then
    cmd+="-itm "
fi
if [ ${use_kl} == True ]; then
    cmd+="-use_kl "
fi
if [ ${use_type} == True ]; then
    cmd+="-use_type "
fi
if [ ${is_debug} == True ]; then
    cmd+="-debug 1 "
fi

echo $cmd
eval $cmd

