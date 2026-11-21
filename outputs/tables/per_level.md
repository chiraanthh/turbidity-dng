**Mean held-out prediction per level (NTU)**

| model | true 0 | true 10 | true 50 | true 200 | true 400 | true 800 |
|---|---|---|---|---|---|---|
| null (train mean) | 128 | 79 | 58 | 44 | 38 | 33 |
| log-log, 1 feature | 42 | 5 | 10 | 45 | 183 | 5041 |
| log-log >=200 + floor | below range | below range | below range | 217 | 358 | 882 |
| ridge, 2 features | 41 | 4 | 10 | 51 | 196 | 3342 |
| GBM, 2 features | 50 | 47 | 0 | 297 | 278 | 355 |
