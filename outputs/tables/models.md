**Leave-one-sample-out CV (6 folds; each fold holds out a whole NTU level)**

| model | MAE (NTU) | RMSE (NTU) | R² | RMSE log10(NTU+1) | chosen |
|---|---|---|---|---|---|
| null (train mean) | 248 | 357 | -0.57 | 1.19 |  |
| log-log, 1 feature | 783 | 1756 | -36.95 | 0.86 |  |
| log-log >=200 + floor | 49 | 67 | 0.94 | 0.79 | ✔ |
| ridge, 2 features | 497 | 1059 | -12.81 | 0.83 |  |
| GBM, 2 features | 147 | 206 | 0.48 | 1.11 |  |
