import matplotlib.pyplot as plt
import numpy as np

from project_paths import DATA_DIR, FIGURES_DIR, ensure_output_dirs

ensure_output_dirs()

# Timestamp [s]    Load cell value    TOF value    Load [N]    TOF [m]
time = []
load = []
displacement = []

with (DATA_DIR / "SparTestData.txt").open("r") as data:
    for line in data:
        splitdata = line.split('	')
        time.append(float(splitdata[0]))
        load.append(float(splitdata[3]) - 160)
        displacement.append(float(splitdata[4]))

time = np.array(time)
load = np.array(load)
displacement = np.array(displacement)

time_full = time.copy()
load_full = load.copy()
displacement_full = displacement.copy() - displacement[0]

mask = time >= 35.0
time = time[mask]
load = load[mask]
displacement = displacement[mask]
displacement = displacement - displacement[0]
time = time - time[0]  # shift so plot starts at 0

# --- Load CV tracking data ---
CV_SCALE = 0.9
CV_TIME_OFFSET = 36.4  # seconds — CV recording starts when load test begins

cv_file = DATA_DIR / "IMG_5710_cv_displacement.txt"
cv_data = np.loadtxt(cv_file, delimiter='\t', skiprows=1)
cv_time = cv_data[:, 0] + CV_TIME_OFFSET
cv_dy   = cv_data[:, 1] * CV_SCALE

twist_data  = np.loadtxt(DATA_DIR / "IMG_5710_cv_twist.txt", delimiter='\t', skiprows=1)
twist_time  = twist_data[:, 0] + CV_TIME_OFFSET
twist_angle = twist_data[:, 1]

# --- Event markers (full dataset time axis) ---
events = [
    (25,  "A"),
    (35,  "B"),
    (160, "C"),
    (174, "D"),
]

def add_event_markers(ax, events):
    _, ymax = ax.get_ylim()
    for t, label in events:
        ax.axvline(x=t, color="black", linewidth=0.8, linestyle="--", zorder=5)
        ax.text(t, ymax, label, fontsize=8, va="top", ha="center", zorder=6,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.6))

events_cd = [(160, "C"), (175, "D")]

# --- Figure 1: Load and Displacement (2 panels) ---
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.subplots_adjust(wspace=0.35)

ax_load = axes1[0]
ax_load.scatter(time_full, load_full, c=time_full, cmap="viridis", s=6)
ax_load.set_xlabel("Time [s]")
ax_load.set_ylabel("Load [N]")
ax_load.axhline(y=1100, color="red", linewidth=1)
ax_load.grid(True)
max_idx = np.argmax(load_full)
t_max   = time_full[max_idx]
_, ymax_load = ax_load.get_ylim()
ax_load.axvline(x=t_max, color="black", linewidth=0.8, linestyle="--", zorder=5)
ymin_load, ymax_load = ax_load.get_ylim()
label_y = ymin_load + (ymax_load - ymin_load) * 0.5
ax_load.text(t_max - 2, label_y, f"Max: {load_full[max_idx]:.0f} N\nt = {t_max:.1f} s",
             fontsize=8, va="center", ha="right", zorder=6,
             bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.6))
add_event_markers(ax_load, events)

ax_disp = axes1[1]
ax_disp.plot(time_full, displacement_full, linewidth=1, color="#1f6eb5")
ax_disp.set_xlabel("Time [s]")
ax_disp.set_ylabel("Displacement [m]")
ax_disp.grid(True)
add_event_markers(ax_disp, events)

fig1.savefig(FIGURES_DIR / "SparTestPlot.png", dpi=150, bbox_inches="tight")

# --- Figure 3: Twist angle over time ---
fig3, ax_twist = plt.subplots(figsize=(8, 5))
ax_twist.scatter(twist_time, twist_angle, color="#7b2fbe", s=6)
ax_twist.set_xlabel("Time [s]")
ax_twist.set_ylabel("Twist angle [°]")
ax_twist.set_xlim(time_full[0], max(time_full[-1], twist_time[-1]))
ax_twist.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=5)
ax_twist.grid(True)
add_event_markers(ax_twist, events_cd)

fig3.savefig(FIGURES_DIR / "SparTestPlot_Twist.png", dpi=150, bbox_inches="tight")

# --- Figure 2: TOF vs CV overlay ---
fig2, ax_ov = plt.subplots(figsize=(8, 5))
ax_ov.set_xlabel("Time [s]")
ax_ov.set_ylabel("Displacement [m]")
ax_ov.grid(True)
ax_ov.plot(time_full, displacement_full, linewidth=1, color="#1f6eb5", label="TOF", zorder=2)
ax_ov.plot(cv_time, cv_dy, linewidth=1.5, color="#b03a2e", label="CV", zorder=3)
ax_ov.legend(fontsize=8, loc="upper left")
add_event_markers(ax_ov, events)

fig2.savefig(FIGURES_DIR / "SparTestPlot_Overlay.png", dpi=150, bbox_inches="tight")

# --- Figure 4: CV only ---
fig4, ax_cv = plt.subplots(figsize=(8, 5))
ax_cv.set_xlabel("Time [s]")
ax_cv.set_ylabel("Displacement [m]")
ax_cv.grid(True)
ax_cv.plot(cv_time - cv_time[0], cv_dy, linewidth=1.5, color="#b03a2e", label="CV")
ax_cv.legend(fontsize=8, loc="upper left")

fig4.savefig(FIGURES_DIR / "SparTestPlot_CV.png", dpi=150, bbox_inches="tight")

plt.show()
