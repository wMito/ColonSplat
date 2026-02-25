# ColonSplat: Reconstruction of Peristaltic Motion in Colonoscopy with Dynamic Gaussian Splatting 
Proof-of-Concept Code 

## Environment

1. Install the Python environment
Please note we use gaussian rasterizer from https://github.com/HKUST-SAIL/RaDe-GS.  
Installation steps were not thouroughly tested yet!  
Commands below should work, however if any module is missing - please install it additionally.
```bash
conda create -n env python=3.9 
conda activate env

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --no-build-isolation submodules/diff-gaussian-rasterization-radegs
pip install --no-build-isolation submodules/simple-knn

```

## To run training, testing and visualisation simply run:
```bash
bash run.sh
bash run_custom_ds.sh
```

## Data:  

All data used in our work is available here: https://zenodo.org/records/18763383


## Acknowledgements:  
We thank the authors of publicly available repositories: 
- [ENDO-4DGX](https://github.com/lastbasket/Endo-4DGX). 
- [EndoPlanar](https://github.com/ThatphumCpre/EndoPlanar)
- [SurgicalGS](https://github.com/neneyork/SurgicalGS)
- [Endo-4DGS](https://github.com/lastbasket/Endo-4DGS)
- [Deform3DGS](https://github.com/jinlab-imvr/Deform3DGS)
- [RADE-GS](https://github.com/HKUST-SAIL/RaDe-GS)


## Implementation notes on fair comparison with baselines
- We noticed differnet baselines have differently implemented LPIPS measurement (input normalization). We pass image in range 0-1 and use normalization inside model. We applied the same computation for all baselines (see how we do it in utils/image_utils.py/lpips_score).  
- For fair comparison of geometry, we do not use densification and pruning for baseline methods on DynamicColon. We mostly do not want to prune Gaussians which could potentially lower the scores CH and HD95 for other baselines. We densely initialize Gaussians with point cloud from Blender which should be sufficient for reconstruction.  