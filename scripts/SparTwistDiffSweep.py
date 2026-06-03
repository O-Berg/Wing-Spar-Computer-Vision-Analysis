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

twist_data      = np.loadtxt(DATA_DIR / "IMG_5710_cv_twist.txt", delimiter='\t', skiprows=1)
twist_time_raw  = twist_data[:, 0]
twist_angle_raw = twist_data[:, 1]

# --- Sweep ---
offsets = np.arange(30.0, 42.0, 0.05)
correlations = []

T_B = 35.0
T_D = 174.0

for offset in offsets:
    cv_time    = cv_time_raw    + offset
    twist_time = twist_time_raw + offset

    t_start = max(time[0], cv_time[0], twist_time[0], T_B)
    t_end   = min(time[-1], cv_time[-1], twist_time[-1], T_D)

    mask = (cv_time >= t_start) & (cv_time <= t_end)
    if mask.sum() < 10:
        correlations.append(np.nan)
        continue

    t_common     = cv_time[mask]
    cv_common    = cv_dy[mask]
    tof_common   = np.interp(t_common, time, displacement)
    twist_common = np.interp(t_common, twist_time, twist_angle_raw)

    diff = tof_common - cv_common
    r = np.corrcoef(twist_common, diff)[0, 1]
    correlations.append(r)

correlations = np.array(correlations)
best_idx = np.nanargmin(correlations)
best_offset = offsets[best_idx]
best_r = correlations[best_idx]
print(f"Best offset: {best_offset:.2f} s  (r = {best_r:.4f})")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(offsets, correlations, linewidth=1.5, color="#7b2fbe")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(best_offset, color="red", linewidth=1, linestyle="--", zorder=5)
ax.text(best_offset, ax.get_ylim()[0], f"{best_offset:.2f} s",
        color="red", fontsize=8, ha="left", va="bottom")
ax.set_title("Twist vs. (TOF − CV) Displacement — Correlation vs. Time Offset")
ax.set_xlabel("CV Time Offset [s]")
ax.set_ylabel("Pearson r")
ax.grid(True)

fig.savefig(FIGURES_DIR / "SparTwistDiffSweep.png", dpi=150, bbox_inches="tight")
plt.show()
