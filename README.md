# guided-brightspot

## Environment Setup

```
conda create -n brightspot python=3.10
conda activate brightspot
pip install monai[all]
pip install SimpleITK
pip install nibabel numpy matplotlib scikit-learn jupyterlab
pip install kaggle
```
Notes:
- `monai[all]` is quite large, set your `PIP_CACHE_DIR` and `TMPDIR` to a directory with enough storage if you have a `No space left on device` error while installing

## Data Setup
Follow the directions in data/download_instructions.md
