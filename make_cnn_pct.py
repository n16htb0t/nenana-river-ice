"""
2D CNN % model (run 18 design: open water / ice / snow / snow-free land % per VIIRS pixel), leave-one-date-out over
the 6 labelled dates, saved so the VIIRS outcome figures can show 2D CNN fractions next to RF and UNet++.
Labels follow the run: box shares (run 19) or Gaussian shares (RIVER_LABELS=psf, run 22).

Outputs: <run output>/pct_models/cnn2d/pct_<date>.tif (4 bands, %), metrics.json

    RIVER_BUFFER=1 [RIVER_LABELS=psf] ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/make_cnn_pct.py
"""

import json

import numpy as np

import rf_viirs_unmix as U
import viirs_river_states as R
import viirs_states as V

OUT = R.OUT / "pct_models" / "cnn2d"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    shifts = {r["date"]: r["shift_fine_px"] for r in json.loads((U.OUT / "results.json").read_text())}
    labelled = [d for d in U.viirs_dates()
                if (U.RUN16 / f"pred_{d}.tif").exists() or (U.RUN16 / "new_dates" / d / "rf_pred.tif").exists()]
    cache = {}
    days = {d: V.load_day(d, shifts.get(d, [0, 0]), cache) for d in labelled}
    res, tp, pp = {"per_date": {}}, [], []
    for test in labelled:
        model = V.fit_unet(days, [d for d in labelled if d != test], arch="cnn2d")
        p = V.predict_unet(model, days[test])
        g = days[test]["good"]
        V.write_pct(OUT, days[test], p)
        res["per_date"][test] = V.part_scores(days[test]["pct"][g], p[g])
        tp.append(days[test]["pct"][g]); pp.append(p[g])
        print(test, {k: round(v, 1) for k, v in res["per_date"][test]["mae_pp"].items()}, flush=True)
    res["all"] = V.part_scores(np.concatenate(tp), np.concatenate(pp))
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))
    print("all", res["all"], "\nwrote", OUT)
