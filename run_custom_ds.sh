#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH='.'

# Shared params -these are our baseline values, the same should be in arguments/__init__.py
DEPTH_W=0.25
DCOL_W=0.005
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
            --iterations $ITERS \
            --densify_from_iter 500000 \
            --pruning_from_iter 500000 \
            --densify_from_iter_fine 500000 \
            --pruning_from_iter_fine 500000 \
            --col_smooth_weight $COL_SMOOTH_W 

        # ---- RENDER ----
        python render.py \
            -s $data_path \
            --model_path output/$exp_name \
            --skip_video --skip_train --eval 

        # ---- METRICS FOR EVERY N-TH TEST FRAME ----
        python metrics.py --model_path output/$exp_name

        # ---- VISUALIZE CUSTOM TEST TRAJECTORY, METRICS FOR TEST TRAJECTORY ----
        python render_custom_test_trajectory.py \
            --model_path output/$exp_name 

        # ---- VISUALIZE TEST LOOKAT CAMERAS ----
        python render_custom_lookat_cameras.py \
            --model_path output/$exp_name 

        # ---- TIME SHIFT ----
        python test_geometry.py \
            -s $data_path \
            --model_path output/$exp_name \
            --skip_video --skip_train --eval 

    done
done
