#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH='.'

# Shared params
DEPTH_W=0.1
DCOL_W=0.05
CENTER_W=0.01
KNN_W=0.01
ITERS=30000

DATASETS=(
    c1_ascending_t4_v4
    c1_cecum_t1_v4
    # c1_descending_t4_v4
    c1_sigmoid1_t4_v4
    c1_sigmoid2_t4_v4
    c1_transverse1_t1_v4
    c1_transverse1_t4_v4
    # c2_cecum_t1_v4
    c2_transverse1_t1_v4
)

for DATA in "${DATASETS[@]}"
do
    echo "===== Running $DATA ====="

    data_path="data/coloncrafter/${DATA}"
    exp_name="runs_no_densification/${DATA}_depth${DEPTH_W}_dcol${DCOL_W}_centerline${CENTER_W}"

    # ---- TRAIN ----
    python train.py \
        -s $data_path \
        --port 6019 \
        --expname $exp_name \
        --no_ds --no_do --no_dr \
        --depth_weight $DEPTH_W \
        --knn_weight $KNN_W \
        --dcol_weight $DCOL_W \
        --centerline_weight $CENTER_W \
        --iterations $ITERS \
        --depth_mode expected \
        --densify_from_iter 500000 \
        --pruning_from_iter 500000 \
        --densify_from_iter_fine 500000 \
        --pruning_from_iter_fine 50000

    ---- RENDER ----
    python render.py \
        -s $data_path \
        --model_path output/$exp_name \
        --skip_video --skip_train \
        --mode monocular --eval \
        --no_ds --no_do --no_dr

    # ---- METRICS ----
    python metrics.py --model_path output/$exp_name

    # ---- VISUALIZE ----
    python render_visualize.py \
        -s $data_path \
        --model_path output/$exp_name \
        --skip_video --skip_train \
        --mode monocular --eval \
        --no_ds --no_do --no_dr

    # ---- TIME SHIFT ----
    python render_time_shift.py \
        -s $data_path \
        --model_path output/$exp_name \
        --skip_video --skip_train \
        --mode monocular --eval \
        --no_ds --no_do --no_dr

done
