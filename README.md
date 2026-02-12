Proof-of-Concept code for Anatomically-Constrained Colonoscopy Reconstruction of Highly Deformable Scenes

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
```
