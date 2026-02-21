#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH='.'

# Shared params -these are our baseline values, the same should be in arguments/__init__.py
DEPTH_W=0.25
DCOL_W=0.005
CENTER_W=0.0
KNN_W=(
    0.01
    )
ITERS=30000
COL_SMOOTH_W=0.0001

DATASETS=(
    data_cecum_v2
    data_rectum
    data_transverse
)

for knn_weight in "${KNN_W[@]}" 
do
    for DATA in "${DATASETS[@]}" 
    do
        echo "===== Running $DATA ====="

        data_path="data_custom/${DATA}"
        exp_name="custom_ds/${DATA}"

        # ---- TRAIN ----
        python train.py \
            -s $data_path \
            --port 6019 \
            --expname $exp_name \
            --depth_weight $DEPTH_W \
            --knn_weight $knn_weight \
            --dcol_weight $DCOL_W \
            --centerline_weight $CENTER_W \
            --iterations $ITERS \
            --densify_from_iter 500000 \
            --pruning_from_iter 500000 \
            --densify_from_iter_fine 500000 \
            --pruning_from_iter_fine 500000 \
            --col_smooth_weight $COL_SMOOTH_W \
            --no_do --use_color_emb #--no_dr --no_ds 

        # ---- RENDER ----
        python render.py \
            -s $data_path \
            --model_path output/$exp_name \
            --skip_video --skip_train --eval \
            --no_do --use_color_emb #--no_ds --no_dr 

        # ---- METRICS ----
        python metrics.py --model_path output/$exp_name

        # ---- VISUALIZE ----
        python render_add_trajectory.py \
            --model_path output/$exp_name \
            --no_do --use_color_emb #--no_dr --no_ds 

        # ---- VISUALIZE CUSTOM TRAJECTORY ----
        python render_lookat_cust.py \
            --model_path output/$exp_name \
            --no_do --use_color_emb #--no_dr --no_ds 

        # ---- TIME SHIFT ----
        python test_chamfer.py \
            -s $data_path \
            --model_path output/$exp_name \
            --skip_video --skip_train --eval \
            --no_do --use_color_emb #--no_dr --no_ds 

    done
done
