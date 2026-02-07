exp_name="debug_code_cleaning"

# PYTHONPATH='.'  python train.py -s data/c1_ascending_t4_v4 --port 6019 --expname $exp_name --no_ds --no_do --no_dr --depth_weight 1.0 --tv_weight 0 --knn_weight 0.01 --dcol_weight 0.05 --iterations 30000

python render.py -s data/c1_ascending_t4_v4  --model_path output/$exp_name --skip_video --skip_train --mode monocular --eval --no_ds --no_do --no_dr
# python metrics.py --model_path output/$exp_name
# python render_visualize.py -s data/c1_ascending_t4_v4  --model_path output/$exp_name --skip_video --skip_train --mode monocular --eval --no_ds --no_do --no_dr
# python render_time_shift.py -s data/c1_ascending_t4_v4  --model_path output/$exp_name --skip_video --skip_train --mode monocular --eval --no_ds --no_do --no_dr


