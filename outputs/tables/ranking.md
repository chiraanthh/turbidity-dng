**Feature ranking (sep = |Δmean| / combined shot sd; >3 ≈ resolvable, optimistic)**

| feature | Spearman ρ (n=6) | p | adjacent pairs in order | sep 50→200 | sep 200→400 | sep 400→800 | shot CV | role | why excluded |
|---|---|---|---|---|---|---|---|---|---|
| decay_len_inner | 0.829 | 0.042 | 4/5 | 6.0 | 2.9 | 5.0 | 1.5 % | SELECTED |  |
| decay_len_halo | 0.829 | 0.042 | 4/5 | 10.8 | 1.6 | 4.3 | 1.2 % | redundant |  |
| core_halo_ratio | -0.829 | 0.042 | 4/5 | 6.8 | 2.8 | 3.8 | 2.5 % | redundant |  |
| max_grad | -0.829 | 0.042 | 4/5 | 4.1 | 2.4 | 6.5 | 4.0 % | excluded (confound) | steepest point sits on the clipped-core edge for 0-50 NTU |
| lap_var | -0.829 | 0.042 | 4/5 | 2.3 | 3.0 | 0.6 | 6.6 % | redundant |  |
| tenengrad | -0.829 | 0.042 | 4/5 | 2.8 | 3.2 | 0.9 | 6.1 % | cross-check |  |
| spec_slope_cclip | -0.943 | 0.005 | 4/5 | 1.8 | 3.3 | 1.5 | 2.2 % | cross-check |  |
| hf_ratio_asis | -0.829 | 0.042 | 4/5 | 15.0 | 1.9 | 1.6 | 16.5 % | excluded (confound) | tracks the clipped-plateau edge (trend vanishes under common clip) |
| f50_cclip | 0.886 | 0.019 | 3/5 | 65.0 | 6.0 | 12.4 | 0.2 % | excluded (confound) | tracks plateau size above T -> core brightness |
| f_noise_asis | -0.943 | 0.005 | 4/5 | 6.3 | 0.6 | 2.9 | 3.7 % | excluded (confound) | signal-to-noise: dimmer image hits noise floor sooner |
