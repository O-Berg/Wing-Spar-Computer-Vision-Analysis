import matplotlib.pyplot as plt
import numpy as np

from project_paths import DATA_DIR, FIGURES_DIR, ensure_output_dirs

ensure_output_dirs()

# --- Load TOF data ---
time = []
displacement = []

with (DATA_DIR / "SparTestData.txt").open("r") as data:
    for line in data:
        splitdata = line.split('	')
        time.append(float(splitdata[0]))
        displacement.append(float(splitdata[4]))

time = np.array(time)
displacement = np.array(displacement)
displacement = displacement - displacement[0]

# --- Load raw CV data ---
CV_SCALE = 0.9

cv_data     = np.loadtxt(DATA_DIR / "IMG_5710_cv_displacement.txt", delimiter='\t', skiprows=1)
cv_time_raw = cv_data[:, 0]
cv_dy       = cv_data[:, 1] * CV_SCALE

# --- Sweep ---
offsets = np.arange(0.0, 50.0, 0.05)
correlations = []

T_B = 35.0
T_C = 160.0

for offset in offsets:
    cv_time = cv_time_raw + offset

    t_start = max(time[0], cv_time[0], T_B)
    t_end   = min(time[-1], cv_time[-1], T_C)

    mask = (cv_time >= t_start) & (cv_time <= t_end)
    if mask.sum() < 10:
        correlations.append(np.nan)
        continue

    t_common   = cv_time[mask]
    cv_common  = cv_dy[mask]
    tof_common = np.interp(t_common, time, displacement)

    r = np.corrcoef(tof_common, cv_common)[0, 1]
    correlations.append(r)

correlations = np.array(correlations)
best_idx = np.nanargmax(correlations)
best_offset = offsets[best_idx]
best_r = correlations[best_idx]
print(f"Best offset: {best_offset:.2f} s  (r = {best_r:.4f})")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(offsets, correlations, linewidth=1.5, color="#1f6eb5")
ax.axvline(best_offset, color="red", linewidth=1, linestyle="--", zorder=5)
ax.text(best_offset, ax.get_ylim()[0], f"{best_offset:.2f} s",
        color="red", fontsize=8, ha="left", va="bottom")
ax.set_title("TOF vs CV Displacement — Correlation vs. Time Offset")
ax.set_xlabel("CV Time Offset [s]")
ax.set_ylabel("Pearson r")
ax.grid(True)

fig.savefig(FIGURES_DIR / "SparOffsetSweep.png", dpi=150, bbox_inches="tight")
plt.show()
