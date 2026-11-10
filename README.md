# daksh — Landsat 30 m river / ice / snow / land models

Random Forest and UNet++ models that label every Landsat 30 m pixel of the Nenana
study box (lon −149.338 … −149.069, lat 64.094 … 64.535), trained only on the
hand-drawn label polygons, which are taken as ground truth. VIIRS comes later; nothing
here uses VIIRS.

Everything is reproducible from the scripts in this folder. Outputs are in `output/`.
**All scores, pixel counts and confusion matrices in one place: `output/RESULTS.md`.**
Each model folder also has a `model_info.txt`: what the model is, its inputs, its training
and test dates and pixels, its scores and confusion matrix.

---

## 1. Data

### Dates and labels

| Date       | Season    | Group (QGIS)  | Labels from                                              | Imagery |
| ---------- | --------- | ------------- | -------------------------------------------------------- | ------- |
| 2023-07-24 | summer    | mathus_naira  | `patches/nenana_20230724/labels/labels_2023-07-24.shp`  | patch `_sr.tif` + `_thermal.tif` |
| 2023-10-04 | freeze-up | mathus_naira  | `patches/nenana_20231004/labels/…`                       | patch |
| 2024-04-21 | breakup   | mathus_naira  | `patches/nenana_20240421/labels/…`                       | patch |
| 2024-06-22 | summer    | mathus_naira  | `patches/nenana_20240622/labels/…`                       | patch |
| 2024-03-12 | winter    | daksh         | `~/labels_20240312.gpkg` (3 polygons)                    | `nenana_crops/20240312/landsat_20240312_box.tif` |
| 2024-04-20, 2024-04-29 | spring | daksh | none yet (predicted only)                           | `nenana_crops/<date>/landsat_*_box.tif` |

`patches/` = `~/Desktop/RiverProject/Matus/narrow_rivers/patches`.
Label classes (`class_id`): 1 river (open water), 2 ice, 3 snow, 4 land. Codes 0 and 9
are ignored. Every date sits on the same 995 × 1635 grid (0.00027027°, EPSG:4326), so
the same pixel row and column is the same place on every date.

### Inputs per pixel

| Dataset          | Inputs | Why |
| ---------------- | ------ | --- |
| `exclude_daksh`  | 11: SR_B2–SR_B7 surface reflectance, ST_B10 surface temperature (K), JRC water occurrence (%), NDWI, NDSI, NDVI | everything the patch images have |
| `with_daksh`     | 9: blue, green, red, NIR, SWIR1, water occurrence, NDWI, NDSI, NDVI | the daksh tiff has bands B1–B6 only (no SWIR2, no temperature), and a model needs the same inputs on every date |
| `external_daksh` | the same 9 | trained on the patch dates only, tested on the daksh labels (section 3) |

The daksh tiff stores Level-1 DN. It is turned into reflectance with the Landsat
Collection 2 constants: `(DN × 2.0e-5 − 0.1) / cos(SZA)`, with SZA read from the tiff
(stored × 100). Pixels with missing imagery are skipped.

### How the dates are combined in training

Only dates with labels are used for training: in `with_daksh` that is the 4 patch dates
+ 2024-03-12; 2024-04-20 and 2024-04-29 have no labels and are only predicted, as test
images. UNet++ input scaling (band means and spreads) is also computed from the
training dates only. All labelled dates are pooled into one training set: one model per task learns from
every date at once, so it sees summer, freeze-up, breakup and (with daksh) winter.
Each pixel is an example: its inputs come from its own date's image, and its label
from its own date's polygons. A date's image is never mixed with another date's labels.
UNet++ draws each training crop from one date at a time.

### Pixels labelled on more than one date

Every date is on the same grid, so polygons of different dates often cover the same
ground. Of the 85,890 labelled pixels in with_daksh, 49,688 locations are labelled on
one date, 13,005 on two and 3,250 on three or more. How that is handled:

- **Training:** each (date, location) is its own example, with that date's inputs and
  that date's label. The same location can be river in July, ice in October and snow in
  March; all are correct and all are used. Nothing is averaged or merged across dates.
- **Cross-validation:** a 64-row band is in the same fold on every date, so a location
  that is tested is held out of training on all dates. A model cannot score well by
  remembering a place from another date.
- **External test:** the 2024-03-12 daksh pixels share 1,108 location-date pairs with
  patch-date labels, often with another class (only 2% match 2023-07-24). It therefore
  tests a new date and season, not new ground.

Shared locations and how often both dates have the same class:

| Dates | Shared px | Same class |
| ----- | --------: | ---------: |
| 2023-07-24 & 2023-10-04 | 2,301 | 29% |
| 2023-07-24 & 2024-03-12 | 327 | 2% |
| 2023-07-24 & 2024-04-21 | 1,955 | 78% |
| 2023-07-24 & 2024-06-22 | 8,710 | 62% |
| 2023-10-04 & 2024-03-12 | 130 | 28% |
| 2023-10-04 & 2024-04-21 | 372 | 74% |
| 2023-10-04 & 2024-06-22 | 7,943 | 76% |
| 2024-03-12 & 2024-04-21 | 241 | 49% |
| 2024-03-12 & 2024-06-22 | 410 | 7% |
| 2024-04-21 & 2024-06-22 | 1,769 | 71% |

---

## 2. Tasks (what a model learns)

| Folder                              | `--task`      | Classes                | From the hand labels |
| ----------------------------------- | ------------- | ---------------------- | -------------------- |
| `four_class_river_ice_snow_land/`   | `four`        | river, ice, snow, land | as drawn |
| `binary/snow_vs_no_snow/`           | `snow`        | no_snow, snow          | snow = ice + snow; no_snow = river + land |
| `binary/river_vs_not_river/`        | `river`       | not_river, river       | river = river + ice; not_river = snow + land |
| `binary/combined_to_four_class/`    | `combined` (no training) | river, ice, snow, land | the two binary answers merged: river&no-snow = river, river&snow = ice, snow off-river = snow, neither = land |
| `one_vs_rest/river_vs_rest/`        | `ind_river`   | not_river, river       | open water vs everything else |
| `one_vs_rest/ice_vs_rest/`          | `ind_ice`     | not_ice, ice           | ice vs everything else |
| `one_vs_rest/snow_vs_rest/`         | `ind_snow`    | not_snow, snow         | snow vs everything else |
| `one_vs_rest/land_vs_rest/`         | `ind_land`    | not_land, land         | land vs everything else |

### Labelled pixels per class

| Dataset | river | ice | snow | land | total |
| ------- | ----: | --: | ---: | ---: | ----: |
| exclude_daksh (train + CV) | 39,991 | 1,732 | 564 | 43,071 | 85,358 |
| with_daksh (train + CV) | 39,991 | 2,074 | 714 | 43,111 | 85,890 |
| external_test_daksh: train | 39,991 | 1,732 | 564 | 43,071 | 85,358 |
| external_test_daksh: test (2024-03-12) | 0 | 342 | 150 | 40 | 532 |

Per date and per task class (e.g. snow vs no snow 2,296 / 83,062): `output/RESULTS.md`.

Why three set-ups: one four-class model must trade the classes off against each other;
two binary models answer the two questions that define the river/land states
(is it frozen, is it snowy); independent one-vs-rest models show how well each class
can be found on its own, which matters for the rare classes (ice 1.7k px, snow 0.6k px
against 40k river and 43k land).

### River/land state mask (your four scene classes)

| id | name                                  | meaning |
| -- | ------------------------------------- | ------- |
| 1  | `ice_free_river_snow_free_land`       | open river, snow-free land |
| 2  | `ice_free_river_snow_covered_land`    | open river, snow-covered land |
| 3  | `ice_covered_river_snow_free_land`    | river frozen, snow-free land |
| 4  | `ice_covered_river_snow_covered_land` | river frozen, snow-covered land |
| 0  | no river in the block                 | |

These need both a river and its surroundings, so a single 30 m pixel cannot have one.
`scene_mask.py` cuts the box into 375 m blocks (12.5 × 12.5 Landsat pixels) and, from a
four-class map: river state = frozen if predicted ice pixels outnumber open-river
pixels; land state = snow-covered if predicted snow outnumbers snow-free land. Blocks
with fewer than 3 river/ice pixels are 0. Every 30 m pixel takes its block's class.

---

## 3. Models

- **Random Forest** (`train_rf.py`): 300 trees, minimum 2 samples per leaf, classes
  weighted by frequency (`balanced_subsample`). Each pixel on its own: no neighbourhood.
- **UNet++** (`train_unetpp.py`): ResNet-34 encoder (ImageNet weights), 1,500 steps of
  8 random 256 × 256 crops centred near labelled pixels, random 90° turns and flips,
  cross-entropy weighted by 1/√(class frequency), AdamW + one-cycle learning rate,
  mixed precision. The loss is only on labelled pixels; unlabelled pixels are context.
  Uses the neighbourhood, so it can use shape and texture (a channel, a forest edge).

### How scores are measured (and why)

5-fold **spatial** cross-validation. The box is cut into east–west bands 64 rows
(about 1.9 km) tall; band *i* belongs to fold *i* mod 5, on every date. Each fold's
model trains on the other 4 folds of all dates and is scored on its own fold, and
training pixels within 12 rows of a test band are dropped. So a pixel is always scored
by a model that never saw it or its polygon's neighbours. A random pixel split would
score pixels whose neighbours from the same polygon were in training, which inflates
the numbers.

After cross-validation each model is retrained once on all labels; that **final model**
draws the prediction maps. Scores in `metrics.txt` are always the cross-validated ones,
never scores on training pixels.

### External test cohort (`external_daksh`, `external_test.py`)

A stricter test: the models are trained on the 4 patch dates only and scored on the
2024-03-12 daksh labels, a date, season (winter) and labeller they never saw. No
cross-validation is needed, since train and test share no date. Same 9 inputs as
`with_daksh`. Only 532 test pixels (342 ice, 150 snow, 40 land), so read the scores as
a check of how the models carry over to a new date, not as a precise number.

Scores cover the labelled areas only. On the unlabelled dates
(`new_dates/`) there is nothing to score against; look at the images.

---

## 4. Output layout

```
output/
├── RESULTS.md                          every pixel count, score and confusion matrix
├── 1_exclude_daksh/                    (--dataset exclude_daksh) 4 patch dates, 11 inputs
│   ├── four_class_river_ice_snow_land/
│   │   ├── random_forest/              model_info.txt, metrics.txt, maps, state masks
│   │   ├── unet_plus_plus/             same + unetpp_final.pt
│   │   ├── evaluation/                 figures comparing the two models
│   │   ├── images/                     JPEGs
│   │   └── new_dates/                  maps of the unlabelled dates
│   ├── binary/
│   │   ├── snow_vs_no_snow/            (same inside)
│   │   ├── river_vs_not_river/
│   │   └── combined_to_four_class/
│   └── one_vs_rest/
│       ├── river_vs_rest/  ice_vs_rest/  snow_vs_rest/  land_vs_rest/
├── 2_with_daksh/                       (--dataset with_daksh) + 2024-03-12 labels, 9 inputs
│   └── (same)
└── 3_external_test_daksh/              (--dataset external_daksh) train patch dates, test 2024-03-12
    └── (same, no new_dates)
```

Inside each task folder:

| Path                                   | What it is |
| -------------------------------------- | ---------- |
| `random_forest/`, `unet_plus_plus/`     | one folder per model |
| `<model>/model_info.txt`                | the model in one page: method and settings, inputs, training/test dates and pixels per class, scores, confusion matrix, files |
| `<model>/metrics.txt`                   | (external_daksh: scores on the daksh labels) cross-validated precision, recall, F1 per class, overall accuracy, macro F1, **confusion matrix** (rows = true, columns = predicted); first for all dates together, then per date; RF also lists input importance |
| `<model>/oof.npz`                       | the cross-validated prediction of every labelled pixel (date, row, col, true, predicted) |
| `<model>/pred_<date>.tif` + `.qml`      | final model's 30 m map for every date of the dataset; the `.qml` gives QGIS the class colours |
| `<model>/scene_mask_<date>.tif`         | (four_class, combined) river/land state mask |
| `<model>/scene_mask_metrics.txt`        | (four_class, combined) state-mask scores on blocks where the hand labels hold both river and land |
| `<model>/scene_classes.csv`             | (four_class, combined) one summary state for the whole box per date; coarse, see the mask instead |
| `<model>/unetpp_final.pt`               | UNet++ final weights + input scaling |
| `evaluation/scores.png`                 | F1, precision, recall per class, RF vs UNet++ |
| `evaluation/confusion.png`              | both confusion matrices, row-normalised, with pixel counts |
| `evaluation/summary.csv`                | every score of both models in one table |
| `evaluation/compare_<date>.png`         | true colour · hand labels · RF · UNet++ for the whole box |
| `evaluation/zoom_<date>.png`            | the same, zoomed on the labelled area |
| `new_dates/<date>/<model>_pred.tif`     | maps for unlabelled dates (2026-03-26 midwinter, 2023-05-05 breakup) from `~/Desktop/Patches` |
| `images/*.jpg`                          | full-resolution JPEGs: image next to image + mask, for every map above (`new_` = unlabelled date, `_state` = river/land state) |

Colours: river blue, ice magenta, snow cyan, land brown, "not …" grey. State mask:
open/snow-free blue, open/snowy light cyan, frozen/snow-free orange, frozen/snowy magenta.

---

## 5. Scripts and how to run

Environment: `~/venvs/river-ml` (PyTorch with CUDA, segmentation-models-pytorch,
scikit-learn, system GDAL). `PY=~/venvs/river-ml/bin/python`.

| Script               | Does |
| -------------------- | ---- |
| `data.py`            | dates, inputs, label mapping per task, spatial folds, score tables, map writing |
| `train_rf.py`        | `--task --dataset`: RF cross-validation + final maps |
| `train_unetpp.py`    | `--task --dataset`: UNet++ cross-validation + final maps (`--iters`, `--encoder`) |
| `combine.py`         | `--dataset`: binary snow + river → four classes, scored on the cross-validated predictions |
| `scene_mask.py`      | `--dataset`: river/land state masks + their scores |
| `evaluate.py`        | `--task --dataset` (`--task combined` too): figures in `evaluation/` |
| `predict_new.py`     | `--task --dataset [--dates]`: maps for unlabelled dates |
| `images.py`          | `--dataset`: all JPEGs |
| `add_to_project.py`  | adds the maps to `all_patches.qgz` (close QGIS first) |
| `run_all.sh`         | four-class + binary, both datasets, + combine + evaluate |
| `run_rest.sh`        | independent models, unlabelled dates, state masks, images |
| `external_test.py`   | `--task`: train on patch dates, score on the daksh labels |
| `run_external.sh`    | external test for every task + combine, state masks, images |
| `summarise.py`       | `model_info.txt` in every model folder + `output/RESULTS.md` |

One model by hand:
```
$PY train_rf.py     --task ind_ice --dataset with_daksh
$PY train_unetpp.py --task ind_ice --dataset with_daksh
$PY evaluate.py     --task ind_ice --dataset with_daksh
```

---

## 6. Things to know when reading the results

- **Rare classes.** Ice (1,732 px) and snow (564 px, 714 with daksh) come from only
  2023-10-04, 2024-04-21 and 2024-03-12, so their scores move a lot with each new polygon.
- **2023-07-24** holds 31,021 of the 39,991 river pixels, so it shapes what the models
  call river. On the unlabelled dates the models mark some vegetated land as river.
- **Clouds** were never labelled, so models call them snow or ice.
- **with_daksh has no temperature input**, which the exclude_daksh Random Forest ranked
  among its 4 most important inputs; compare the two datasets with that in mind.
- **Per-date macro F1** only averages classes present on that date (a summer date has
  no ice or snow).
