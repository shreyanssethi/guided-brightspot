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

## Preprocessing Steps

1. Run the following to apply resampling, cropping, binarizing, and normaling fixes to the raw data (Basically, the issues we found in `data_exploration.ipynb`) 
`python preprocess_rawData.py --split training --split test  --verify`

2. Run the following to actually compute the classical segmentation results (for comparison) and the soft maps that will be fed into the U-Net:
`python preprocessing/compute_soft_maps.py --split training --split test`

## Notebook Breakdown
All notebooks can be found in the /notebooks subdirectory. Here is a breakdown of each one:
- `data_exploration.ipynb` --> Analyzes the raw data, prints statistics on T1/FLAIR shaping/sizes by testing site, checks what pre-processing is necessary
- `verify_processed.ipynb` --> Verify the outputs of running `preprocessing/preprocess_rawData.py`. Check shapes/sizes match, images align, and no cases of zero WMH voxels