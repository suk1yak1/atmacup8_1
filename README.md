# 8.1 atmaCup [playground] Coffee Review

- LightGBM
- StratifiedKFold(n_splits=10) by binned total_cup_points

## Environment
```bash
pip install poetry
poetry install
```

## Run
``` bash
cd src
poetry run python preprocessing.py
poetry run python train.py
```
