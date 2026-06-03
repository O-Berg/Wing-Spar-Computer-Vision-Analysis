import matplotlib.pyplot as plt
import numpy as np

from project_paths import DATA_DIR, FIGURES_DIR, ensure_output_dirs

ensure_output_dirs()

# --- Load TOF data ---
time_raw = []
load_raw = []
displacement_raw = []

with (DATA_DIR / "SparTestData.txt").open("r") as data:
    for line in data:
        splitdata = line.split('	')
        time_raw.append(float(splitdata[0]))
        load_raw.append(float(splitdata[3]) - 160)
        displacement_raw.append(float(splitdata[4]))

time_full        = np.array(time_raw)
load_full        = np.array(load_raw)
displacement_full = np.array(displacement_raw) - displacement_raw[0]

# --- Load CV data ---
CV_SCALE      = 0.85
CV_TIME_OFFSET = 36.4

cv_data  = np.loadtxt(DATA_DIR / "IMG_5710_cv_displacement.txt", delimiter='\t', skiprows=1)
cv_time  = cv_data[:, 0] + CV_TIME_OFFSET
cv_dy    = cv_data[:, 1] * CV_SCALE

# --- Find best vertical shift for TOF to match CV over B–C ---
T_B_val, T_C_val = 35.0, 160.0
mask_bc   = (cv_time >= T_B_val) & (cv_time <= T_C_val)
t_bc      = cv_time[mask_bc]
cv_bc     = cv_dy[mask_bc]
tof_bc    = np.interp(t_bc, time_full, displacement_full)
tof_shift = np.mean(cv_bc - tof_bc)
displacement_full = displacement_full + tof_shift
print(f"TOF vertical shift applied: {tof_shift:.4f} m")

twist_data  = np.loadtxt(DATA_DIR / "IMG_5710_cv_twist.txt", delimiter='\t', skiprows=1)
twist_time  = twist_data[:, 0] + CV_TIME_OFFSET
twist_angle = twist_data[:, 1]

# --- Offset sweep (B–C window) ---
cv_time_raw = cv_data[:, 0]
offsets      = np.arange(0.0, 50.0, 0.05)
correlations = []
T_B, T_C, T_D = 35.0, 160.0, 174.0

for offset in offsets:
    ct = cv_time_raw + offset
    t_start = max(time_full[0], ct[0], T_B)
    t_end   = min(time_full[-1], ct[-1], T_C)
    mask = (ct >= t_start) & (ct <= t_end)
    if mask.sum() < 10:
        correlations.append(np.nan)
        continue
    tof_c = np.interp(ct[mask], time_full, displacement_full)
    r = np.corrcoef(tof_c, cv_dy[mask])[0, 1]
    correlations.append(r)

correlations = np.array(correlations)
best_idx    = np.nanargmax(correlations)
best_offset = offsets[best_idx]
best_r      = correlations[best_idx]

# --- Event marker helper ---
events_all = [(25, "A"), (35, "B"), (160, "C"), (174, "D")]
events_cd  = [(160, "C"), (174, "D")]

def add_markers(ax, events):
    _, ymax = ax.get_ylim()
    for t, label in events:
        ax.axvline(x=t, color="black", linewidth=0.8, linestyle="--", zorder=5)
        ax.text(t, ymax, label, fontsize=8, va="top", ha="center", zorder=6,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.6))

# ======================================================================
# Figure 1: Load only
# ======================================================================
fig1, ax_load = plt.subplots(figsize=(9, 5))
fig1.subplots_adjust(top=0.95)

ax_load.scatter(time_full, load_full, c=time_full, cmap="viridis", s=6)
ax_load.axhline(y=1100, color="red", linewidth=1)
ax_load.set_xlabel("Time [s]")
ax_load.set_ylabel("Load [N]")
ax_load.grid(True)
max_idx = np.argmax(load_full)
t_max   = time_full[max_idx]
ax_load.axvline(x=t_max, color="black", linewidth=0.8, linestyle="--", zorder=5)
ymin_load, ymax_load = ax_load.get_ylim()
label_y = ymin_load + (ymax_load - ymin_load) * 0.5
ax_load.text(t_max - 2, label_y, f"Max: {load_full[max_idx]:.0f} N\nt = {t_max:.1f} s",
             fontsize=8, va="center", ha="right", zorder=6,
             bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.6))
add_markers(ax_load, events_all)

fig1.savefig(FIGURES_DIR / "ReportFinal_Load.png", dpi=150, bbox_inches="tight")

# ======================================================================
# Figure 2: TOF and CV side by side
# ======================================================================
fig2, (ax_tof, ax_cv) = plt.subplots(1, 2, figsize=(16, 5))
fig2.subplots_adjust(wspace=0.35, top=0.95)

ax_tof.plot(time_full, displacement_full, linewidth=1, color="#1f6eb5", label="TOF")
ax_tof.set_xlabel("Time [s]")
ax_tof.set_ylabel("Displacement [m]")
ax_tof.legend(fontsize=8, loc="upper left")
ax_tof.grid(True)
add_markers(ax_tof, events_all)

ax_cv.plot(cv_time, cv_dy, linewidth=1.5, color="#b03a2e", label="CV")
ax_cv.set_xlabel("Time [s]")
ax_cv.set_ylabel("Displacement [m]")
ax_cv.legend(fontsize=8, loc="upper left")
ax_cv.grid(True)

fig2.savefig(FIGURES_DIR / "ReportFinal_Displacement.png", dpi=150, bbox_inches="tight")

# ======================================================================
# Figure 3: Offset sweep + overlay side by side
# ======================================================================
fig3, (ax_sweep, ax_ov) = plt.subplots(1, 2, figsize=(16, 5))
fig3.subplots_adjust(wspace=0.35, top=0.95)

ax_sweep.plot(offsets, correlations, linewidth=1.5, color="#1f6eb5")
ax_sweep.axvline(best_offset, color="red", linewidth=1, linestyle="--", zorder=5)
ax_sweep.text(best_offset, ax_sweep.get_ylim()[0],
              f"{best_offset:.2f} s", color="red", fontsize=8, ha="left", va="bottom")
ax_sweep.set_xlabel("CV Time Offset [s]")
ax_sweep.set_ylabel("Pearson r")
ax_sweep.grid(True)

ax_ov.plot(time_full, displacement_full, linewidth=1, color="#1f6eb5", label="TOF", zorder=2)
ax_ov.plot(cv_time, cv_dy, linewidth=1.5, color="#b03a2e", label="CV", zorder=3)
ax_ov.set_xlabel("Time [s]")
ax_ov.set_ylabel("Displacement [m]")
ax_ov.legend(fontsize=8, loc="upper left")
ax_ov.grid(True)
add_markers(ax_ov, events_all)

fig3.savefig(FIGURES_DIR / "ReportFinal_SweepOverlay.png", dpi=150, bbox_inches="tight")

# ======================================================================
# Figure 4: Twist angle over time
# ======================================================================
fig4, ax_twist = plt.subplots(figsize=(9, 5))
fig4.subplots_adjust(top=0.95)

ax_twist.scatter(twist_time, twist_angle, color="#7b2fbe", s=6, label="Twist")
ax_twist.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=5)
ax_twist.set_xlim(time_full[0], max(time_full[-1], twist_time[-1]))
ax_twist.set_xlabel("Time [s]")
ax_twist.set_ylabel("Twist angle [°]")
ax_twist.legend(fontsize=8, loc="upper left")
ax_twist.grid(True)
add_markers(ax_twist, events_cd)

fig4.savefig(FIGURES_DIR / "ReportFinal_Twist.png", dpi=150, bbox_inches="tight")

# ======================================================================
# Figure 5: Noise std comparison between TOF and CV in B–C window
# ======================================================================
# Interpolate both signals onto a common time grid within 50–90s
T_NOISE_START, T_NOISE_END = 50.0, 90.0
mask_tof_bc = (time_full >= T_NOISE_START) & (time_full <= T_NOISE_END)
t_common = time_full[mask_tof_bc]

tof_bc_seg = displacement_full[mask_tof_bc]
cv_bc_seg  = np.interp(t_common, cv_time, cv_dy)

# Detrend each segment by subtracting a rolling-average trend
def detrend(t, y, window_sec=5.0):
    dt  = np.mean(np.diff(t))
    win = max(3, int(round(window_sec / dt)) | 1)  # odd window
    half = win // 2
    y_padded = np.pad(y, half, mode='reflect')
    trend = np.convolve(y_padded, np.ones(win) / win, mode='valid')
    return y - trend, trend

tof_fs = 1.0 / np.mean(np.diff(t_common))
cv_fs  = 1.0 / np.mean(np.diff(cv_time[(cv_time >= T_NOISE_START) & (cv_time <= T_NOISE_END)]))
print(f"TOF sample rate: {tof_fs:.2f} Hz  ({len(t_common)} samples over {T_NOISE_END - T_NOISE_START:.0f} s)")
print(f"CV  sample rate: {cv_fs:.2f} Hz  ({((cv_time >= T_NOISE_START) & (cv_time <= T_NOISE_END)).sum()} samples over {T_NOISE_END - T_NOISE_START:.0f} s)")

tof_noise, tof_trend = detrend(t_common, tof_bc_seg)
cv_noise,  cv_trend  = detrend(t_common, cv_bc_seg)

tof_std = np.std(tof_noise)
cv_std  = np.std(cv_noise)

fig5, (ax_noise_sig, ax_noise_bar) = plt.subplots(1, 2, figsize=(14, 5))
fig5.subplots_adjust(wspace=0.35, top=0.95)

ax_noise_sig.plot(t_common, tof_noise, linewidth=0.8, color="#1f6eb5",
                  label=f"TOF  σ = {tof_std*1e3:.3f} mm", zorder=2)
ax_noise_sig.plot(t_common, cv_noise, linewidth=0.8, color="#b03a2e",
                  label=f"CV   σ = {cv_std*1e3:.3f} mm", zorder=3)
ax_noise_sig.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax_noise_sig.set_xlabel("Time [s]")
ax_noise_sig.set_ylabel("Detrended displacement [m]")
ax_noise_sig.set_title("Noise in 50–90s window (detrended)")
ax_noise_sig.legend(fontsize=8, loc="upper right")
ax_noise_sig.grid(True)

labels = ["TOF", "CV"]
stds   = [tof_std * 1e3, cv_std * 1e3]
colors = ["#1f6eb5", "#b03a2e"]
bars = ax_noise_bar.bar(labels, stds, color=colors, width=0.4)
for bar, val in zip(bars, stds):
    ax_noise_bar.text(bar.get_x() + bar.get_width() / 2,
                      val + max(stds) * 0.02,
                      f"{val:.3f} mm", ha="center", va="bottom", fontsize=9)
ax_noise_bar.set_ylabel("Noise std [mm]")
ax_noise_bar.set_title("Noise std comparison (50–90s)")
ax_noise_bar.grid(True, axis="y")
ax_noise_bar.set_ylim(0, max(stds) * 1.25)

fig5.savefig(FIGURES_DIR / "ReportFinal_NoiseStd.png", dpi=150, bbox_inches="tight")

# ======================================================================
# Figure 6: Raw B–C signals with linear trend overlaid
# ======================================================================
fig6, ax_tr = plt.subplots(figsize=(10, 5))
fig6.subplots_adjust(top=0.95)

ax_tr.plot(t_common, tof_trend,  linewidth=3.0, color="orange",    label="Rolling trend (5 s)", zorder=2)
ax_tr.plot(t_common, cv_trend,   linewidth=3.0, color="gold",      label="CV trend (5 s)",      zorder=2)
ax_tr.plot(t_common, tof_bc_seg, linewidth=0.8, color="#1f6eb5",   label="TOF signal",          zorder=3)
ax_tr.plot(t_common, cv_bc_seg,  linewidth=0.8, color="#b03a2e",   label="CV signal",           zorder=3)
ax_tr.set_xlabel("Time [s]")
ax_tr.set_ylabel("Displacement [m]")
ax_tr.set_title("TOF & CV – 50 to 90s with rolling trends (window=5.0 s)")
ax_tr.legend(fontsize=8, loc="upper left")
ax_tr.grid(True)

fig6.savefig(FIGURES_DIR / "ReportFinal_TrendCheck.png", dpi=150, bbox_inches="tight")


plt.show()
