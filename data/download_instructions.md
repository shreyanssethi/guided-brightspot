# Data Download

The dataset is the [WMH 2017 Challenge](https://www.kaggle.com/datasets/farahmo/wmh-dataset) hosted on Kaggle.

## 1. Get a Kaggle API token

Go to kaggle.com → Settings → API → **Create New Token**. This downloads `kaggle.json` with your username and key.

## 2. Download the dataset

```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_api_key
cd data/
kaggle datasets download -d farahmo/wmh-dataset --unzip
```

This creates `data/wmh_data/` with `training/` (60 cases, 3 sites) and `test/` (110 cases, 5 sites) subdirectories.

## 3. Next step

Run preprocessing from the repo root:
```bash
python preprocessing/preprocess_rawData.py --split training --split test --verify
```
This creates `data/processed/` with standardized images ready for training.
