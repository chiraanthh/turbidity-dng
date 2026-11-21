**Robustness: change in predicted NTU (one shot per sample)**

| test | level | median abs Δpred % (≥200 NTU) | worst abs Δpred % | fit band intact (worst) | floor-class flips |
|---|---|---|---|---|---|
| exposure | 0.5 | 0.0 | 0.0 | 100 % | 0 |
| exposure | 0.7 | 0.0 | 0.0 | 100 % | 0 |
| exposure | 1.4 | 0.0 | 0.0 | 90 % | 0 |
| exposure | 2.0 | 0.7 | 0.9 | 60 % | 0 |
| noise | 1.0 | 0.0 | 0.1 | 100 % | 0 |
| noise | 2.0 | 0.0 | 0.1 | 100 % | 0 |
| noise | 4.0 | 0.0 | 0.3 | 100 % | 0 |
| shift | 5.0 | 1.3 | 2.4 | 100 % | 0 |
| shift | 10.0 | 2.9 | 4.7 | 100 % | 0 |
| shift | 20.0 | 4.7 | 10.3 | 100 % | 0 |
| shift | 40.0 | 11.4 | 23.5 | 97 % | 0 |
| shift | 80.0 | 23.9 | 57.4 | 73 % | 6 |
