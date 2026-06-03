import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

# --- Load CV data ---
CV_SCALE = 0.85
CV_TIME_OFFSET = 36.4

cv_data     = np.loadtxt(DATA_DIR / "IMG_5710_cv_displacement.txt", delimiter='\t', skiprows=1)
cv_time_raw = cv_data[:, 0]
cv_dy       = cv_data[:, 1] * CV_SCALE

twist_data      = np.loadtxt(DATA_DIR / "IMG_5710_cv_twist.txt", delimiter='\t', skiprows=1)
twist_time_raw  = twist_data[:, 0]
twist_angle_raw = twist_data[:, 1]

# --- Compute common data ---
cv_time    = cv_time_raw    + CV_TIME_OFFSET
twist_time = twist_time_raw + CV_TIME_OFFSET

t_start = max(time[0], cv_time[0], twist_time[0])
t_end   = min(time[-1], cv_time[-1], twist_time[-1])

mask         = (cv_time >= t_start) & (cv_time <= t_end)
t_common     = cv_time[mask]
cv_common    = cv_dy[mask]
tof_common   = np.interp(t_common, time, displacement)
twist_common = np.interp(t_common, twist_time, twist_angle_raw)

# --- Vertical shift TOF to match CV over B–C ---
T_B, T_C = 35.0, 160.0
mask_bc   = (t_common >= T_B) & (t_common <= T_C)
tof_shift = np.mean(cv_common[mask_bc] - tof_common[mask_bc])
tof_common = tof_common + tof_shift

diff = tof_common - cv_common

r = np.corrcoef(twist_common, diff)[0, 1]
print(f"TOF vertical shift: {tof_shift:.4f} m  |  Pearson r: {r:.4f}")

# --- Fits ---
angle_sorted = np.linspace(twist_common.min(), twist_common.max(), 300)

# -|A * sin(θ - B)| + C
def abs_sin_model(x, A, B, C):
    return -np.abs(A * np.sin(np.deg2rad(x - B))) + C

popt, _ = curve_fit(abs_sin_model, twist_common, diff,
                    p0=[abs(diff.min()), 0.0, 0.0], maxfev=10000)
abs_sin_fit = abs_sin_model(angle_sorted, *popt)
ss_res = np.sum((diff - abs_sin_model(twist_common, *popt))**2)
ss_tot = np.sum((diff - diff.mean())**2)
r2_sin = 1 - ss_res / ss_tot
sin_label = (f"-|{popt[0]:.4f} · sin(θ - {popt[1]:.2f}°)| + {popt[2]:.4f}"
             f"  (R² = {r2_sin:.3f})")

# --- Figure 1: Twist, diff, scatter ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.subplots_adjust(wspace=0.4)
ax_twist, ax_diff, ax_corr = axes

ax_twist.scatter(t_common, twist_common, color="#7b2fbe", s=4)
ax_twist.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=5)
ax_twist.set_xlabel("Time [s]")
ax_twist.set_ylabel("Twist angle [°]")
ax_twist.grid(True)

ax_diff.plot(t_common, diff, linewidth=1, color="#2e7d32")
ax_diff.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=5)
ax_diff.set_xlabel("Time [s]")
ax_diff.set_ylabel("Difference [m]")
ax_diff.grid(True)

sc = ax_corr.scatter(twist_common, diff, c=t_common, cmap="plasma", s=6, zorder=2)
ax_corr.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
ax_corr.axvline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
ax_corr.set_xlabel("Twist angle [°]")
ax_corr.set_ylabel("TOF − CV [m]")
cbar = fig.colorbar(sc, ax=ax_corr, pad=0.02, fraction=0.046)
cbar.set_label("Time [s]", fontsize=8)
ax_corr.grid(True)

fig.savefig(FIGURES_DIR / "SparDiffPlot.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "ReportFinal_DiffCorr.png", dpi=150, bbox_inches="tight")

# --- Figure 2: Sine fit ---
fig2, ax_sin = plt.subplots(figsize=(8, 5))

ax_sin.scatter(twist_common, diff, c=t_common, cmap="plasma", s=6, zorder=2)
ax_sin.plot(angle_sorted, abs_sin_fit, color="#b03a2e", linewidth=1.5, zorder=4, label=sin_label)
ax_sin.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
ax_sin.axvline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
ax_sin.set_xlabel("Twist angle [°]")
ax_sin.set_ylabel("TOF − CV [m]")
ax_sin.legend(fontsize=7, loc="upper left", framealpha=0.9)
ax_sin.grid(True)
sm2 = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(t_common[0], t_common[-1]))
cbar2 = fig2.colorbar(sm2, ax=ax_sin, pad=0.02, fraction=0.046)
cbar2.set_label("Time [s]", fontsize=8)

fig2.savefig(FIGURES_DIR / "SparDiffPlot_Fits.png", dpi=150, bbox_inches="tight")
fig2.savefig(FIGURES_DIR / "ReportFinal_DiffFits.png", dpi=150, bbox_inches="tight")
plt.show()
