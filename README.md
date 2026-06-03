# Wing Spar Analysis

Python scripts and source data for comparing wing spar load test data with computer vision displacement and twist tracking.

## Project Contents

- `scripts/` contains the analysis and plotting scripts.
- `data/` contains the load, TOF, and extracted computer vision tracking data.
- `config/` contains saved tracker calibration and ROI setup.
- `figures/` contains generated analysis plots.
- `raw/` is for local raw video captures that should not be committed.

The raw capture video `raw/IMG_5710.mov` is intentionally ignored by git because it is about 984 MB. Keep it locally in `raw/` when running `analysis.py`, or pass another video path with `--video`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Generate the final report figures:

```bash
python scripts/ReportFinal.py
```

Run the interactive tracker:

```bash
python scripts/analysis.py --video raw/IMG_5710.mov
```

## Report Figures

| Load response | TOF and CV displacement |
| --- | --- |
| ![Load response](figures/ReportFinal_Load.png) | ![TOF and CV displacement](figures/ReportFinal_Displacement.png) |

| Offset sweep and overlay | Twist angle |
| --- | --- |
| ![Offset sweep and displacement overlay](figures/ReportFinal_SweepOverlay.png) | ![Twist angle](figures/ReportFinal_Twist.png) |

| TOF vs CV comparison | Noise standard deviation |
| --- | --- |
| ![TOF vs CV comparison](figures/ReportFinal_TOF_vs_CV.png) | ![Noise standard deviation](figures/ReportFinal_NoiseStd.png) |

| Trend check | Difference correlation |
| --- | --- |
| ![Trend check](figures/ReportFinal_TrendCheck.png) | ![Difference correlation](figures/ReportFinal_DiffCorr.png) |

| Difference fit |
| --- |
| ![Difference fit](figures/ReportFinal_DiffFits.png) |

## Notes

Generated `.png` figures are kept in the repository so the latest analysis outputs can be viewed without rerunning the scripts. Raw video files and local Python cache files are excluded.
