# ConvLSTM screening (validation only)

Step 4 of `.claude/plans/convlstm.md`. Every setup was trained with 2 members on four pooled all-crop splits
(`europe/{loyo,loro}/all`, `south_america/{loyo,loro}/all`) and scored on **val**; test scores were not used for any
choice. "Field R2" is per country × crop subset, mean ± spread over the 2 members; the summary row takes the median
over the 20 subsets. The World splits are not part of the screening; they are trained only in the final run.

Status 2026-10-22: screening complete. Chosen for the final run: ConvLSTM base on the 10-day cube with a 3-step
(30-day) time shift (`convlstm_w10`).

## Takeaways so far

- **Model setups:** no ConvLSTM variant beats the patch LSTM (`lstm_patch_v3_s2`) on the original 30-day cube
  (median field R2 0.262). Base is the best ConvLSTM (0.235); the 9×9 patch (`k9`) is about as good but 2.6× slower
  and less stable; two layers (`l2`) are similar; the wider model (`h64`) is worst. Differences are mostly within the
  member spread. Base was kept.
- **Data setups** (all on ConvLSTM base): denser time series help. Median field R2 0.299 with 10-day windows and
  0.283 with 15-day windows, vs 0.235 on the original cube; median correlation rises from 0.64 to about 0.71. Gains
  are uneven: Argentina, Switzerland and Brazil soybean (LORO) improve, while Brazil corn and Uruguay/Brazil soybean
  (LOYO) get worse with 10-day windows.
- A milder time shift on the 10-day cube (1 step = 10 days, `convlstm_w10_s1`) is a tie: median 0.294 vs 0.299,
  correlation 0.67 vs 0.71. The 30-day shift (`convlstm_w10`) was kept.
- The 30-day rebuild (0.193) scores below the original cube (0.235). It starts at day 210 (T = 17 instead of 24), so
  it drops early windows that are empty for most fields, and differs from the original in 410 Brazilian pixels.
- Subsets with few val fields (6–15) are noisy; read them loosely.

## Setups

| config | change from `convlstm_base` |
|---|---|
| `convlstm_base` | 5×5 patch, 32 hidden channels, 1 layer, 3×3 kernel, original 30-day cube (24 steps), time shift 1 step |
| `convlstm_k9` | 9×9 patch |
| `convlstm_h64` | 64 hidden channels |
| `convlstm_l2` | 2 layers |
| `convlstm_w30` | 30-day cube rebuilt from raw scenes, from day 210 (17 steps) |
| `convlstm_w15` | 15-day cube (34 steps), time shift 2 steps (30 days) |
| `convlstm_w10` | 10-day cube (51 steps), time shift 3 steps (30 days) |
| `convlstm_w10_s1` | 10-day cube, time shift 1 step (10 days) |

## Model setups (original cube)

"Above reference" counts subsets where the setup beats `lstm_patch_v3_s2`.

### Summary

| config | subsets | median field R2 | median member spread | above reference | median corr | median offset share | median within-field corr |
|---|---|---|---|---|---|---|---|
| lstm_patch_v3_s2 | 20 | 0.262 | 0.073 | 0/20 | 0.63 | 0.14 | 0.46 |
| convlstm_base | 20 | 0.235 | 0.058 | 8/20 | 0.64 | 0.12 | 0.47 |
| convlstm_k9 | 20 | 0.227 | 0.088 | 11/20 | 0.57 | 0.05 | 0.46 |
| convlstm_h64 | 20 | 0.168 | 0.036 | 9/20 | 0.64 | 0.12 | 0.47 |
| convlstm_l2 | 20 | 0.226 | 0.094 | 8/20 | 0.65 | 0.13 | 0.46 |

### Field R2 per subset (mean ± spread over members)

| split | subset | fields | lstm_patch_v3_s2 | convlstm_base | convlstm_k9 | convlstm_h64 | convlstm_l2 |
|---|---|---|---|---|---|---|---|
| europe/loro/all | Germany / rapeseed | 30 | 0.14 ± 0.20 | -0.09 ± 0.01 | 0.15 ± 0.08 | 0.03 ± 0.03 | 0.07 ± 0.09 |
| europe/loro/all | Germany / wheat | 10 | -0.55 ± 0.28 | -0.31 ± 0.11 | -0.23 ± 0.23 | -0.71 ± 0.09 | -0.78 ± 0.11 |
| europe/loro/all | Switzerland / wheat | 19 | -0.22 ± 0.27 | -0.08 ± 0.18 | -0.01 ± 0.01 | -0.16 ± 0.12 | -0.22 ± 0.07 |
| europe/loyo/all | Germany / rapeseed | 15 | -0.48 ± 0.04 | -0.91 ± 0.08 | -0.06 ± 0.47 | -0.51 ± 0.05 | -0.36 ± 0.26 |
| europe/loyo/all | Germany / wheat | 30 | 0.27 ± 0.02 | 0.33 ± 0.06 | 0.36 ± 0.14 | 0.16 ± 0.13 | 0.06 ± 0.02 |
| europe/loyo/all | Switzerland / wheat | 7 | 0.57 ± 0.07 | 0.58 ± 0.04 | 0.54 ± 0.25 | 0.75 ± 0.01 | 0.54 ± 0.10 |
| europe/loyo/all | Switzerland / winter_cereal | 6 | 0.52 ± 0.01 | 0.25 ± 0.18 | 0.58 ± 0.05 | 0.46 ± 0.02 | 0.34 ± 0.18 |
| south_america/loro/all | Argentina / corn | 31 | 0.44 ± 0.14 | 0.52 ± 0.03 | 0.41 ± 0.14 | 0.49 ± 0.02 | 0.42 ± 0.00 |
| south_america/loro/all | Argentina / soybean | 79 | 0.82 ± 0.03 | 0.78 ± 0.00 | 0.80 ± 0.02 | 0.80 ± 0.00 | 0.80 ± 0.01 |
| south_america/loro/all | Argentina / wheat | 8 | 0.66 ± 0.01 | 0.63 ± 0.02 | 0.63 ± 0.05 | 0.70 ± 0.01 | 0.70 ± 0.02 |
| south_america/loro/all | Brazil / corn | 26 | 0.63 ± 0.04 | 0.39 ± 0.21 | 0.17 ± 0.24 | 0.53 ± 0.17 | 0.56 ± 0.08 |
| south_america/loro/all | Brazil / soybean | 69 | 0.05 ± 0.11 | -0.12 ± 0.01 | 0.27 ± 0.02 | 0.13 ± 0.01 | 0.19 ± 0.09 |
| south_america/loro/all | Brazil / wheat | 40 | 0.06 ± 0.34 | 0.36 ± 0.00 | 0.03 ± 0.27 | 0.18 ± 0.03 | 0.21 ± 0.10 |
| south_america/loro/all | Uruguay / soybean | 59 | 0.02 ± 0.01 | 0.06 ± 0.06 | 0.12 ± 0.01 | 0.07 ± 0.00 | 0.07 ± 0.02 |
| south_america/loyo/all | Argentina / corn | 70 | 0.46 ± 0.01 | 0.46 ± 0.05 | 0.45 ± 0.07 | 0.48 ± 0.03 | 0.57 ± 0.00 |
| south_america/loyo/all | Argentina / soybean | 78 | 0.41 ± 0.07 | 0.35 ± 0.09 | 0.44 ± 0.10 | 0.42 ± 0.08 | 0.41 ± 0.12 |
| south_america/loyo/all | Argentina / wheat | 13 | 0.33 ± 0.06 | 0.22 ± 0.02 | 0.61 ± 0.02 | 0.21 ± 0.04 | 0.38 ± 0.30 |
| south_america/loyo/all | Brazil / soybean | 43 | 0.26 ± 0.19 | 0.06 ± 0.13 | -0.33 ± 0.49 | 0.09 ± 0.43 | 0.24 ± 0.12 |
| south_america/loyo/all | Brazil / wheat | 21 | -5.32 ± 0.69 | -6.43 ± 4.03 | -5.99 ± 4.66 | -6.08 ± 1.98 | -5.92 ± 0.00 |
| south_america/loyo/all | Uruguay / soybean | 86 | 0.10 ± 0.11 | 0.21 ± 0.11 | 0.19 ± 0.05 | -0.01 ± 0.14 | 0.07 ± 0.18 |

## Data setups (ConvLSTM base)

"Above reference" counts subsets where the setup beats `convlstm_base`.

### Summary

| config | subsets | median field R2 | median member spread | above reference | median corr | median offset share | median within-field corr |
|---|---|---|---|---|---|---|---|
| lstm_patch_v3_s2 | 20 | 0.262 | 0.073 | 12/20 | 0.63 | 0.14 | 0.46 |
| convlstm_base | 20 | 0.235 | 0.058 | 0/20 | 0.64 | 0.12 | 0.47 |
| convlstm_w30 | 20 | 0.193 | 0.071 | 11/20 | 0.63 | 0.12 | 0.47 |
| convlstm_w15 | 20 | 0.283 | 0.072 | 13/20 | 0.72 | 0.12 | 0.51 |
| convlstm_w10 | 20 | 0.299 | 0.056 | 13/20 | 0.71 | 0.12 | 0.52 |
| convlstm_w10_s1 | 20 | 0.294 | 0.049 | 13/20 | 0.67 | 0.14 | 0.50 |

### Field R2 per subset (mean ± spread over members)

| split | subset | fields | lstm_patch_v3_s2 | convlstm_base | convlstm_w30 | convlstm_w15 | convlstm_w10 | convlstm_w10_s1 |
|---|---|---|---|---|---|---|---|---|
| europe/loro/all | Germany / rapeseed | 30 | 0.14 ± 0.20 | -0.09 ± 0.01 | 0.12 ± 0.05 | 0.18 ± 0.04 | 0.21 ± 0.04 | 0.33 ± 0.00 |
| europe/loro/all | Germany / wheat | 10 | -0.55 ± 0.28 | -0.31 ± 0.11 | -0.62 ± 0.24 | -0.47 ± 0.08 | -0.19 ± 0.07 | -0.07 ± 0.04 |
| europe/loro/all | Switzerland / wheat | 19 | -0.22 ± 0.27 | -0.08 ± 0.18 | 0.03 ± 0.12 | 0.04 ± 0.12 | 0.34 ± 0.05 | 0.17 ± 0.06 |
| europe/loyo/all | Germany / rapeseed | 15 | -0.48 ± 0.04 | -0.91 ± 0.08 | -0.52 ± 0.12 | -0.55 ± 0.05 | -0.80 ± 0.28 | -0.69 ± 0.39 |
| europe/loyo/all | Germany / wheat | 30 | 0.27 ± 0.02 | 0.33 ± 0.06 | 0.41 ± 0.01 | 0.05 ± 0.19 | 0.16 ± 0.02 | 0.16 ± 0.28 |
| europe/loyo/all | Switzerland / wheat | 7 | 0.57 ± 0.07 | 0.58 ± 0.04 | 0.14 ± 0.10 | 0.09 ± 0.60 | 0.31 ± 0.15 | 0.34 ± 0.15 |
| europe/loyo/all | Switzerland / winter_cereal | 6 | 0.52 ± 0.01 | 0.25 ± 0.18 | 0.47 ± 0.18 | 0.66 ± 0.07 | 0.70 ± 0.00 | 0.61 ± 0.04 |
| south_america/loro/all | Argentina / corn | 31 | 0.44 ± 0.14 | 0.52 ± 0.03 | 0.46 ± 0.03 | 0.63 ± 0.05 | 0.67 ± 0.00 | 0.67 ± 0.02 |
| south_america/loro/all | Argentina / soybean | 79 | 0.82 ± 0.03 | 0.78 ± 0.00 | 0.81 ± 0.03 | 0.72 ± 0.04 | 0.79 ± 0.01 | 0.83 ± 0.02 |
| south_america/loro/all | Argentina / wheat | 8 | 0.66 ± 0.01 | 0.63 ± 0.02 | 0.61 ± 0.03 | 0.80 ± 0.01 | 0.71 ± 0.08 | 0.85 ± 0.03 |
| south_america/loro/all | Brazil / corn | 26 | 0.63 ± 0.04 | 0.39 ± 0.21 | 0.38 ± 0.04 | 0.32 ± 0.05 | 0.15 ± 0.01 | 0.26 ± 0.03 |
| south_america/loro/all | Brazil / soybean | 69 | 0.05 ± 0.11 | -0.12 ± 0.01 | 0.18 ± 0.06 | 0.34 ± 0.02 | 0.28 ± 0.02 | 0.25 ± 0.19 |
| south_america/loro/all | Brazil / wheat | 40 | 0.06 ± 0.34 | 0.36 ± 0.00 | 0.02 ± 0.10 | 0.31 ± 0.08 | 0.36 ± 0.16 | 0.38 ± 0.12 |
| south_america/loro/all | Uruguay / soybean | 59 | 0.02 ± 0.01 | 0.06 ± 0.06 | -0.01 ± 0.08 | 0.08 ± 0.10 | 0.02 ± 0.04 | -0.01 ± 0.03 |
| south_america/loyo/all | Argentina / corn | 70 | 0.46 ± 0.01 | 0.46 ± 0.05 | 0.48 ± 0.00 | 0.54 ± 0.04 | 0.52 ± 0.06 | 0.43 ± 0.00 |
| south_america/loyo/all | Argentina / soybean | 78 | 0.41 ± 0.07 | 0.35 ± 0.09 | 0.54 ± 0.05 | 0.49 ± 0.06 | 0.66 ± 0.02 | 0.68 ± 0.03 |
| south_america/loyo/all | Argentina / wheat | 13 | 0.33 ± 0.06 | 0.22 ± 0.02 | 0.26 ± 0.17 | 0.42 ± 0.11 | 0.72 ± 0.13 | 0.33 ± 0.17 |
| south_america/loyo/all | Brazil / soybean | 43 | 0.26 ± 0.19 | 0.06 ± 0.13 | 0.17 ± 0.13 | 0.26 ± 0.19 | -0.21 ± 0.15 | -0.23 ± 0.40 |
| south_america/loyo/all | Brazil / wheat | 21 | -5.32 ± 0.69 | -6.43 ± 4.03 | -8.05 ± 2.90 | -5.05 ± 0.10 | -5.89 ± 2.20 | -6.07 ± 1.24 |
| south_america/loyo/all | Uruguay / soybean | 86 | 0.10 ± 0.11 | 0.21 ± 0.11 | 0.21 ± 0.02 | 0.20 ± 0.07 | -0.19 ± 0.24 | -0.01 ± 0.06 |
