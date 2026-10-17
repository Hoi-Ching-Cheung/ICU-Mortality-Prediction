# Data

Not committed: the raw files are about 90 MB, and the splits plus Bayesian traces add about 600 MB.

**Raw data:** WiDS Datathon 2020 on [Kaggle](https://www.kaggle.com/competitions/widsdatathon2020/data). Put these files here:

```
data/
  training_v2.csv
  WiDS Datathon 2020 Dictionary.csv
```

**Generated** (by the notebooks, in order):

| Path | Made by |
|---|---|
| `lasso_coeff.csv`, `lasso_score.csv` | `03_modeling.ipynb` §5 (Lasso path) |
| `preprocessed/12-01/*.csv` | `02_preprocessing.ipynb` |
| `bayesian_model_checkpoints/*` | `03_modeling.ipynb` §6.4 (slow; needs GPU) |
