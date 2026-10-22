# Sentinel-2 scene availability per season

Written by `scripts/LSTM/12_scene_availability.py`. Grid: 30-day windows from day 0, 24 steps
(day 0 = 1 January of the year before the harvest year, as the Sentinel-2 cube in `lstm/data`). Per-window data:
`/scratch2/oscipal/yieldsat_work/scene_availability/windows_w30.parquet`.

Reproduce:

```bash
source /scratch2/oscipal/cropyield/bin/activate
python scripts/LSTM/12_scene_availability.py --window-days 30 --start-day 0 --n-times 24
```

## How to read

- A window is **in season** when it overlaps sowing..harvest (`seeding_date`, `harvesting_date` of the field's
  metadata). Only in-season windows are counted below.
- **valid share** of a scene = share of the field's labelled pixels with SCL 4 or 5 (vegetation, bare soil); only
  these pixels get values in the cube. The **chosen** scene is the one the YieldSAT rule puts in the cube: fewest
  SCL 8 + 9 (cloud) pixels on the whole raster, ties to the latest. **best** = the highest valid share of any scene
  in the window.
- Classes: **no_scene** (no Sentinel-2 scene in the window), **all_unusable** (best < 5 %), **rule_loss** (chosen
  < 5 % although best >= 50 %: the rule threw away a good scene), **valid** (chosen >= 50 %), **partly_valid** (the
  rest: chosen 5–50 %, or chosen < 5 % with best 5–50 %).
- **usable** = the chosen scene has >= 5 % valid pixels on the field, i.e. the model sees something in that window.
- **dominant cause** = the largest non-valid SCL group on the field in the chosen scene: cloud (8, 9), cirrus (10),
  shadow (3), snow (11), dark/unclassified (2, 7; mostly haze and low sun), nodata (0, 1), water (6).
- Shares are % of in-season windows. All numbers are aggregates; Switzerland is included only as aggregates.

## Headline

86 % of 16,151 in-season windows (2,246 field-years) have a usable
observation (chosen >= 5 %); with a perfect selection rule it would be 91 %
(best >= 5 %).

The rule loses a usable scene (chosen < 5 %, best >= 5 %) in 725 windows (4.49 %);
the strict rule_loss class (best >= 50 %) has 584 windows (3.62 %) in 470
field-years.

In the rule_loss windows the chosen scene is mostly lost to cirrus 72 %, snow 7 %, nodata 7 %, dark 7 %, shadow 7 %
(share of rule_loss windows by dominant cause of the chosen scene). The rule counts only SCL 8 + 9 on the whole
raster, so a scene with cirrus (10), haze (2, 7) or snow over the field, or clouds only over the field, can win.
A "most SCL 4/5 pixels on the field" rule (a variant of `11_build_s2_grid.py`) would raise the usable share from
86 % to 91 % and the valid share (>= 50 %) from 82 % to
87 %.

Dominant cause in windows without a usable chosen scene: cloud 36 %, cirrus 28 %, dark 19 %, snow 9 %, shadow 4 %, nodata 2 %, water 1 %.

| country | crop | field-years | in-season windows | scenes / window | no_scene % | usable % (chosen >= 5 %) | valid % (chosen >= 50 %) | best >= 5 % (perfect rule) | best >= 50 % | rule_loss % |
|---|---|---|---|---|---|---|---|---|---|---|
| Argentina | corn | 185 | 1377 | 7.0 | 0 | 94 | 92 | 98 | 98 | 5 |
| Argentina | soybean | 440 | 2721 | 8.1 | 1 | 94 | 92 | 98 | 98 | 4 |
| Argentina | wheat | 126 | 863 | 8.3 | 0 | 95 | 94 | 99 | 98 | 4 |
| Brazil | corn | 118 | 748 | 5.5 | 0 | 96 | 92 | 98 | 97 | 2 |
| Brazil | soybean | 293 | 1621 | 4.8 | 5 | 88 | 85 | 91 | 88 | 2 |
| Brazil | wheat | 140 | 815 | 5.1 | 5 | 92 | 90 | 93 | 93 | 1 |
| Germany | rapeseed | 111 | 1368 | 6.5 | 3 | 68 | 58 | 77 | 67 | 7 |
| Germany | wheat | 188 | 2116 | 8.6 | 3 | 62 | 51 | 72 | 59 | 6 |
| Switzerland | wheat | 38 | 389 | 5.2 | 1 | 68 | 63 | 76 | 71 | 5 |
| Switzerland | winter_cereal | 35 | 358 | 4.6 | 1 | 70 | 65 | 74 | 69 | 3 |
| Uruguay | soybean | 572 | 3775 | 6.4 | 0 | 96 | 94 | 98 | 97 | 2 |

## Per country × crop × month of the window start

Columns are % of the in-season windows that start in that month.

| country | crop | month | windows | no_scene | all_unusable | rule_loss | partly_valid | valid | dominant cause (unusable) |
|---|---|---|---|---|---|---|---|---|---|
| Argentina | corn | Jan | 185 | 0 | 0 | 5 | 2 | 93 | cirrus (90 %) |
| Argentina | corn | Feb | 176 | 0 | 2 | 2 | 6 | 90 | cirrus (57 %) |
| Argentina | corn | Mar | 141 | 0 | 0 | 1 | 4 | 96 | nodata (100 %) |
| Argentina | corn | Apr | 103 | 0 | 0 | 12 | 2 | 86 | cirrus (83 %) |
| Argentina | corn | May | 96 | 0 | 0 | 11 | 2 | 86 | cirrus (100 %) |
| Argentina | corn | Jun | 73 | 0 | 0 | 3 | 3 | 95 | cirrus (50 %) |
| Argentina | corn | Jul | 37 | 3 | 8 | 8 | 3 | 78 | cirrus (100 %) |
| Argentina | corn | Aug | 66 | 0 | 3 | 2 | 2 | 94 | cloud (67 %) |
| Argentina | corn | Sep | 89 | 0 | 0 | 3 | 2 | 94 | cirrus (100 %) |
| Argentina | corn | Oct | 101 | 0 | 0 | 5 | 1 | 94 | cirrus (100 %) |
| Argentina | corn | Nov | 137 | 1 | 0 | 7 | 2 | 91 | cirrus (100 %) |
| Argentina | corn | Dec | 173 | 2 | 6 | 1 | 0 | 92 | cloud (91 %) |
| Argentina | soybean | Jan | 440 | 0 | 0 | 4 | 2 | 94 | cirrus (72 %) |
| Argentina | soybean | Feb | 440 | 0 | 0 | 3 | 1 | 95 | cirrus (69 %) |
| Argentina | soybean | Mar | 415 | 2 | 1 | 5 | 5 | 87 | cirrus (76 %) |
| Argentina | soybean | Apr | 153 | 4 | 5 | 8 | 3 | 81 | cirrus (47 %) |
| Argentina | soybean | May | 8 | 12 | 38 | 0 | 0 | 50 | cloud (100 %) |
| Argentina | soybean | Sep | 93 | 8 | 11 | 2 | 1 | 78 | cloud (58 %) |
| Argentina | soybean | Oct | 308 | 0 | 0 | 1 | 1 | 98 | cirrus (100 %) |
| Argentina | soybean | Nov | 424 | 0 | 0 | 6 | 3 | 92 | cirrus (71 %) |
| Argentina | soybean | Dec | 440 | 1 | 0 | 1 | 3 | 95 | cirrus (50 %) |
| Argentina | wheat | Apr | 36 | 0 | 3 | 0 | 0 | 97 | cloud (100 %) |
| Argentina | wheat | May | 114 | 0 | 3 | 1 | 4 | 93 | cirrus (40 %) |
| Argentina | wheat | Jun | 126 | 0 | 1 | 1 | 3 | 95 | cirrus (75 %) |
| Argentina | wheat | Jul | 126 | 0 | 0 | 11 | 4 | 85 | cirrus (93 %) |
| Argentina | wheat | Aug | 126 | 0 | 0 | 1 | 0 | 99 | nodata (100 %) |
| Argentina | wheat | Sep | 126 | 0 | 0 | 6 | 1 | 93 | cirrus (88 %) |
| Argentina | wheat | Oct | 126 | 0 | 0 | 2 | 2 | 97 | cirrus (100 %) |
| Argentina | wheat | Nov | 83 | 1 | 0 | 5 | 2 | 92 | cirrus (100 %) |
| Brazil | corn | Jan | 112 | 0 | 1 | 6 | 6 | 87 | cirrus (100 %) |
| Brazil | corn | Feb | 117 | 0 | 1 | 0 | 9 | 90 | cirrus (100 %) |
| Brazil | corn | Mar | 93 | 0 | 12 | 1 | 2 | 85 | cloud (92 %) |
| Brazil | corn | Apr | 77 | 0 | 0 | 0 | 0 | 100 |  |
| Brazil | corn | May | 73 | 0 | 0 | 7 | 5 | 88 | cirrus (100 %) |
| Brazil | corn | Jun | 64 | 0 | 0 | 0 | 2 | 98 |  |
| Brazil | corn | Jul | 37 | 0 | 0 | 0 | 0 | 100 |  |
| Brazil | corn | Aug | 10 | 0 | 0 | 0 | 20 | 80 |  |
| Brazil | corn | Sep | 11 | 0 | 0 | 0 | 0 | 100 |  |
| Brazil | corn | Oct | 44 | 0 | 0 | 0 | 11 | 89 |  |
| Brazil | corn | Nov | 44 | 0 | 0 | 0 | 0 | 100 |  |
| Brazil | corn | Dec | 66 | 0 | 0 | 2 | 0 | 98 | cirrus (100 %) |
| Brazil | soybean | Jan | 258 | 0 | 4 | 0 | 4 | 92 | cloud (90 %) |
| Brazil | soybean | Feb | 167 | 3 | 1 | 0 | 3 | 93 | cirrus (100 %) |
| Brazil | soybean | Mar | 22 | 0 | 0 | 0 | 0 | 100 |  |
| Brazil | soybean | Aug | 93 | 4 | 0 | 0 | 0 | 96 |  |
| Brazil | soybean | Sep | 206 | 14 | 2 | 0 | 2 | 83 | cloud (100 %) |
| Brazil | soybean | Oct | 293 | 15 | 0 | 5 | 5 | 75 | cirrus (89 %) |
| Brazil | soybean | Nov | 293 | 0 | 11 | 2 | 3 | 85 | cloud (67 %) |
| Brazil | soybean | Dec | 289 | 0 | 8 | 3 | 9 | 80 | cirrus (52 %) |
| Brazil | wheat | Mar | 63 | 0 | 11 | 0 | 5 | 84 | cloud (100 %) |
| Brazil | wheat | Apr | 140 | 0 | 0 | 1 | 4 | 94 | cirrus (100 %) |
| Brazil | wheat | May | 140 | 0 | 0 | 1 | 1 | 98 | cirrus (100 %) |
| Brazil | wheat | Jun | 140 | 0 | 0 | 0 | 0 | 100 |  |
| Brazil | wheat | Jul | 140 | 0 | 0 | 1 | 0 | 99 | shadow (100 %) |
| Brazil | wheat | Aug | 137 | 17 | 0 | 5 | 2 | 76 | cirrus (100 %) |
| Brazil | wheat | Sep | 55 | 38 | 11 | 0 | 4 | 47 | cloud (50 %) |
| Germany | rapeseed | Jan | 111 | 0 | 19 | 5 | 19 | 57 | cloud (57 %) |
| Germany | rapeseed | Feb | 111 | 0 | 32 | 3 | 6 | 59 | cloud (77 %) |
| Germany | rapeseed | Mar | 111 | 5 | 9 | 13 | 5 | 68 | cirrus (65 %) |
| Germany | rapeseed | Apr | 111 | 0 | 2 | 5 | 5 | 88 | cirrus (75 %) |
| Germany | rapeseed | May | 111 | 0 | 6 | 4 | 3 | 87 | cirrus (55 %) |
| Germany | rapeseed | Jun | 111 | 0 | 9 | 10 | 11 | 70 | cloud (36 %) |
| Germany | rapeseed | Jul | 147 | 18 | 27 | 0 | 14 | 41 | cloud (92 %) |
| Germany | rapeseed | Aug | 111 | 0 | 9 | 8 | 8 | 75 | cloud (33 %) |
| Germany | rapeseed | Sep | 111 | 9 | 30 | 14 | 8 | 39 | cloud (58 %) |
| Germany | rapeseed | Oct | 111 | 0 | 17 | 12 | 28 | 43 | snow (26 %) |
| Germany | rapeseed | Nov | 111 | 0 | 25 | 6 | 30 | 39 | dark (53 %) |
| Germany | rapeseed | Dec | 111 | 0 | 54 | 5 | 12 | 30 | cloud (61 %) |
| Germany | wheat | Jan | 188 | 0 | 30 | 5 | 22 | 43 | dark (55 %) |
| Germany | wheat | Feb | 188 | 0 | 14 | 9 | 15 | 62 | cloud (51 %) |
| Germany | wheat | Mar | 188 | 1 | 10 | 14 | 6 | 69 | cirrus (67 %) |
| Germany | wheat | Apr | 188 | 0 | 1 | 7 | 2 | 90 | cirrus (67 %) |
| Germany | wheat | May | 188 | 0 | 6 | 6 | 4 | 84 | cirrus (52 %) |
| Germany | wheat | Jun | 188 | 0 | 3 | 6 | 9 | 81 | cirrus (55 %) |
| Germany | wheat | Jul | 159 | 13 | 14 | 2 | 9 | 62 | cloud (69 %) |
| Germany | wheat | Aug | 92 | 24 | 28 | 0 | 5 | 42 | cloud (92 %) |
| Germany | wheat | Sep | 173 | 7 | 32 | 10 | 14 | 37 | cloud (45 %) |
| Germany | wheat | Oct | 188 | 0 | 45 | 5 | 28 | 22 | dark (48 %) |
| Germany | wheat | Nov | 188 | 0 | 62 | 1 | 31 | 6 | dark (60 %) |
| Germany | wheat | Dec | 188 | 0 | 60 | 3 | 28 | 9 | dark (47 %) |
| Switzerland | wheat | Jan | 38 | 0 | 26 | 8 | 3 | 63 | cloud (62 %) |
| Switzerland | wheat | Feb | 38 | 0 | 11 | 5 | 18 | 66 | cirrus (36 %) |
| Switzerland | wheat | Mar | 38 | 0 | 3 | 11 | 3 | 84 | cirrus (100 %) |
| Switzerland | wheat | Apr | 38 | 0 | 5 | 3 | 3 | 89 | cirrus (100 %) |
| Switzerland | wheat | May | 38 | 0 | 0 | 0 | 0 | 100 |  |
| Switzerland | wheat | Jun | 38 | 0 | 0 | 3 | 3 | 95 | cirrus (100 %) |
| Switzerland | wheat | Jul | 10 | 10 | 30 | 0 | 0 | 60 | cloud (100 %) |
| Switzerland | wheat | Aug | 9 | 33 | 56 | 0 | 0 | 11 | cloud (80 %) |
| Switzerland | wheat | Sep | 28 | 0 | 29 | 4 | 4 | 64 | cloud (78 %) |
| Switzerland | wheat | Oct | 38 | 0 | 39 | 8 | 11 | 42 | cloud (68 %) |
| Switzerland | wheat | Nov | 38 | 0 | 66 | 0 | 8 | 26 | cloud (60 %) |
| Switzerland | wheat | Dec | 38 | 0 | 47 | 16 | 21 | 16 | dark (36 %) |
| Switzerland | winter_cereal | Jan | 35 | 0 | 34 | 0 | 9 | 57 | cloud (50 %) |
| Switzerland | winter_cereal | Feb | 35 | 0 | 20 | 3 | 11 | 66 | cirrus (58 %) |
| Switzerland | winter_cereal | Mar | 35 | 0 | 0 | 17 | 0 | 83 | cirrus (100 %) |
| Switzerland | winter_cereal | Apr | 35 | 0 | 0 | 0 | 0 | 100 |  |
| Switzerland | winter_cereal | May | 35 | 0 | 0 | 0 | 0 | 100 |  |
| Switzerland | winter_cereal | Jun | 35 | 0 | 0 | 6 | 3 | 91 | snow (100 %) |
| Switzerland | winter_cereal | Jul | 7 | 14 | 0 | 0 | 0 | 86 |  |
| Switzerland | winter_cereal | Aug | 4 | 50 | 0 | 0 | 0 | 50 |  |
| Switzerland | winter_cereal | Sep | 32 | 0 | 25 | 0 | 12 | 62 | cloud (62 %) |
| Switzerland | winter_cereal | Oct | 35 | 0 | 60 | 0 | 14 | 26 | cloud (86 %) |
| Switzerland | winter_cereal | Nov | 35 | 0 | 80 | 0 | 6 | 14 | cloud (71 %) |
| Switzerland | winter_cereal | Dec | 35 | 0 | 40 | 3 | 6 | 51 | cloud (67 %) |
| Uruguay | soybean | Jan | 572 | 0 | 0 | 2 | 1 | 98 | cirrus (60 %) |
| Uruguay | soybean | Feb | 572 | 0 | 0 | 3 | 2 | 95 | cirrus (88 %) |
| Uruguay | soybean | Mar | 568 | 0 | 1 | 2 | 3 | 95 | cirrus (67 %) |
| Uruguay | soybean | Apr | 315 | 1 | 17 | 5 | 5 | 72 | cloud (71 %) |
| Uruguay | soybean | May | 32 | 34 | 19 | 0 | 9 | 38 | cloud (83 %) |
| Uruguay | soybean | Oct | 572 | 0 | 0 | 1 | 1 | 98 | cirrus (83 %) |
| Uruguay | soybean | Nov | 572 | 0 | 0 | 1 | 1 | 98 | nodata (71 %) |
| Uruguay | soybean | Dec | 572 | 0 | 0 | 3 | 2 | 94 | cirrus (59 %) |
