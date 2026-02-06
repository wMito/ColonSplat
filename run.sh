
exp_name="debug_all_dep0.025_clamp0.5_knn0.005"

PYTHONPATH='.'  python train.py -s data/c1_ascending_t4_v4 --port 6019 --expname $exp_name --no_ds --no_do  --tv_weight 0 --depth_weight 0.025 --knn_weight 0.005 --iterations 30000

python render.py -s data/c1_ascending_t4_v4  --model_path output/$exp_name --skip_video --skip_train --mode monocular --eval --no_ds --no_do
python metrics.py --model_path output/$exp_name
python render_visualize.py -s data/c1_ascending_t4_v4  --model_path output/$exp_name --skip_video --skip_train --mode monocular --eval --no_ds --no_do


