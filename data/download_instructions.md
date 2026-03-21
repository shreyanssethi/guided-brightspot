Go to kaggle.com --> Log in --> Settings --> API --> "Create New Token"

```
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_api_token_here
cd data/
kaggle datasets download -d farahmo/wmh-dataset --unzip
```

This should generate a wmh_data/ subdirectory with the training/ and test/ folders.