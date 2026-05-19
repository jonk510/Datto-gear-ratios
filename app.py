"""
Gearbox Comparison — Interactive Streamlit App

Overlay two gearbox+diff combinations on a sawtooth shift diagram,
matching the style of the static PNGs produced by gearbox_analysis.py.
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

TYRE_WIDTH_MM     = 205
TYRE_ASPECT_RATIO = 0.50
RIM_DIAMETER_INCH = 15

RPM_MIN            = 1000
RPM_MAX            = 8000
RPM_REDLINE        = 7500
RPM_POWERBAND_LOW  = 4500
RPM_POWERBAND_HIGH = 7500

DIFF_RATIOS = {
    "3.540:1": 3.540,
    "3.700:1": 3.700,
    "3.900:1": 3.900,
    "4.110:1": 4.110,
    "4.375:1": 4.375,
}

GEARBOXES = {
    "SR20 5-Speed (FS5R90A)": {
        "short": "SR20",
        "ratios": {
            "1st": 3.321,
            "2nd": 2.077,
            "3rd": 1.308,
            "4th": 1.000,
            "5th": 0.759,
        },
    },
    "ZF 8HP50 8-Speed": {
        "short": "ZF",
        "ratios": {
            "1st": 4.714,
            "2nd": 3.143,
            "3rd": 2.106,
            "4th": 1.667,
            "5th": 1.285,
            "6th": 1.000,
            "7th": 0.839,
            "8th": 0.667,
        },
    },
    "Nismo/HPI S15 6-Speed (FS6R31A)": {
        "short": "Nismo",
        "ratios": {
            "1st": 3.083,
            "2nd": 1.888,
            "3rd": 1.330,
            "4th": 1.034,
            "5th": 0.834,
            "6th": 0.667,
        },
    },
}

ENGINE_TORQUE_NM = {
    2000: 195.0,
    2500: 205.0,
    3000: 215.0,
    3500: 240.0,
    4000: 285.0,
    4500: 330.0,
    5000: 365.0,
    5500: 395.0,
    6000: 410.0,
    6125: 412.5,
    6500: 405.0,
    6790: 390.0,
    7000: 370.0,
    7200: 350.0,
}
DRIVETRAIN_EFF = 0.95

# Combo A — bold, saturated (solid lines)
COLORS_A = [
    "#1f77b4", "#17becf", "#2ca02c", "#9467bd",
    "#3399ff", "#006688", "#33aa77", "#6655cc",
]
# Combo B — warm reds/oranges (dashed lines)
COLORS_B = [
    "#d62728", "#ff7f0e", "#c05020", "#cc6600",
    "#aa2200", "#884400", "#cc4400", "#993300",
]

# ---------------------------------------------------------------------------
# CORE MATHS
# ---------------------------------------------------------------------------

def tyre_circumference_m(width_mm, aspect, rim_inch):
    sidewall_mm = width_mm * aspect
    diameter_mm = (rim_inch * 25.4) + (2 * sidewall_mm)
    return math.pi * diameter_mm / 1000


def speed_kmh(rpm, gear_ratio, diff_ratio, circ_m):
    wheel_rpm = rpm / (gear_ratio * diff_ratio)
    return wheel_rpm * circ_m * 60 / 1000


def rpm_at_speed(speed_kmh_val, gear_ratio, diff_ratio, circ_m):
    wheel_rpm = (speed_kmh_val * 1000) / (circ_m * 60)
    return wheel_rpm * gear_ratio * diff_ratio


def engine_torque_nm(rpm):
    rpms    = sorted(ENGINE_TORQUE_NM.keys())
    torques = [ENGINE_TORQUE_NM[r] for r in rpms]
    return float(np.interp(rpm, rpms, torques))


def wheel_torque_nm(rpm, gear_ratio, diff_ratio):
    return engine_torque_nm(rpm) * gear_ratio * diff_ratio * DRIVETRAIN_EFF

# ---------------------------------------------------------------------------
# DRAWING HELPERS
# ---------------------------------------------------------------------------

def draw_sawtooth(ax, gearbox, diff, colors, linewidth, linestyle,
                  combo_label, suffix, shift_rpm, circ_m,
                  start_rpm=1500, start_gear_idx=0,
                  origin_dotted=False):
    """Draw sawtooth RPM-vs-speed lines for one gearbox+diff combo."""
    ratios = list(gearbox["ratios"].values())
    gears  = list(gearbox["ratios"].keys())

    entry_rpm = start_rpm
    for k in range(start_gear_idx):
        entry_rpm = shift_rpm * ratios[k + 1] / ratios[k]

    for i in range(start_gear_idx, len(ratios)):
        gear  = gears[i]
        ratio = ratios[i]
        col   = colors[i % len(colors)]

        spd_entry = speed_kmh(entry_rpm, ratio, diff, circ_m)
        spd_shift = speed_kmh(shift_rpm,  ratio, diff, circ_m)

        # Faint extension from origin to entry speed
        origin_ls = (0, (1, 6)) if origin_dotted else "-"
        ax.plot([0, spd_entry], [0, entry_rpm],
                color="lightgrey", linewidth=0.9,
                linestyle=origin_ls, alpha=0.45, zorder=1)

        # Rising diagonal
        is_first_gear = (i == start_gear_idx)
        ax.plot([spd_entry, spd_shift], [entry_rpm, shift_rpm],
                color=col, linewidth=linewidth, linestyle=linestyle,
                label=f"{combo_label} {gear}" if is_first_gear else f"{combo_label} {gear}",
                solid_capstyle="round", zorder=2)

        # Gear label box near the midpoint of the diagonal
        mid_spd = (spd_entry + spd_shift) / 2
        mid_rpm = (entry_rpm + shift_rpm) / 2
        ax.text(mid_spd, mid_rpm + 170,
                f"{gear}\n{suffix}",
                fontsize=7.5, color=col, ha="center",
                fontweight="bold",
                fontstyle="italic" if origin_dotted else "normal",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", alpha=0.6,
                          edgecolor=col, linewidth=0.6),
                zorder=4)

        # Vertical drop to next gear
        if i < len(ratios) - 1:
            next_entry = shift_rpm * ratios[i + 1] / ratio
            ax.plot([spd_shift, spd_shift], [shift_rpm, next_entry],
                    color=col, linewidth=linewidth * 0.65,
                    linestyle=linestyle, alpha=0.7)
            entry_rpm = next_entry


def draw_torque(ax2, gearbox, diff, colors, shift_rpm, circ_m,
                start_rpm=1500, start_gear_idx=0,
                linewidth=2.0, alpha=0.85, fill_alpha=0.08,
                linestyle="--"):
    """Draw wheel torque curve on secondary axis for one gearbox+diff combo."""
    ratios = list(gearbox["ratios"].values())

    entry_rpm = start_rpm
    for k in range(start_gear_idx):
        entry_rpm = shift_rpm * ratios[k + 1] / ratios[k]

    x_all, y_all = [], []

    for i in range(start_gear_idx, len(ratios)):
        ratio   = ratios[i]
        col     = colors[i % len(colors)]
        rpm_pts = np.linspace(entry_rpm, shift_rpm, 200)
        spd_pts = np.array([speed_kmh(r, ratio, diff, circ_m) for r in rpm_pts])
        wt_pts  = np.array([wheel_torque_nm(r, ratio, diff) for r in rpm_pts])

        ax2.fill_between(spd_pts, wt_pts, 0,
                         alpha=fill_alpha, color=col, zorder=2)
        x_all.extend(spd_pts)
        y_all.extend(wt_pts)

        if i < len(ratios) - 1:
            next_entry = shift_rpm * ratios[i + 1] / ratio
            next_wt    = wheel_torque_nm(next_entry, ratios[i + 1], diff)
            x_all.append(spd_pts[-1])
            y_all.append(next_wt)
            entry_rpm = next_entry

    line_col = colors[start_gear_idx % len(colors)]
    ax2.plot(x_all, y_all, color=line_col,
             linewidth=linewidth, linestyle=linestyle,
             alpha=alpha, zorder=3)

# ---------------------------------------------------------------------------
# STATS HELPER
# ---------------------------------------------------------------------------

def combo_stats(gearbox, diff, circ_m):
    ratios = list(gearbox["ratios"].values())
    gears  = list(gearbox["ratios"].keys())
    top_speed  = speed_kmh(RPM_REDLINE, ratios[-1], diff, circ_m)
    rpm_100    = rpm_at_speed(100, ratios[-1], diff, circ_m)
    rpm_110    = rpm_at_speed(110, ratios[-1], diff, circ_m)
    overall_1st = ratios[0] * diff
    overall_top = ratios[-1] * diff
    return {
        "Top gear":        gears[-1],
        "Top speed (km/h)": f"{top_speed:.1f}",
        "Cruise @ 100 km/h": f"{rpm_100:,.0f} rpm",
        "Cruise @ 110 km/h": f"{rpm_110:,.0f} rpm",
        "Overall 1st ratio": f"{overall_1st:.3f}:1",
        f"Overall {gears[-1]} ratio": f"{overall_top:.3f}:1",
    }

# ---------------------------------------------------------------------------
# PAGE LAYOUT
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Gearbox Comparison", layout="wide")
st.title("Gearbox Sawtooth Comparison")
st.caption(f"Tyre: {TYRE_WIDTH_MM}/{int(TYRE_ASPECT_RATIO*100)}R{RIM_DIAMETER_INCH}  |  Redline: {RPM_REDLINE:,} rpm  |  Drivetrain eff: {DRIVETRAIN_EFF:.0%}")

circ_m        = tyre_circumference_m(TYRE_WIDTH_MM, TYRE_ASPECT_RATIO, RIM_DIAMETER_INCH)
gearbox_names = list(GEARBOXES.keys())
diff_labels   = list(DIFF_RATIOS.keys())

# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Chart settings")
    shift_rpm = st.slider("Shift RPM", 5000, RPM_REDLINE, 7200, 100,
                          help="Engine RPM at which each upshift occurs")
    x_max = st.slider("X-axis max speed (km/h)", 120, 320, 240, 10)
    show_powerband = st.checkbox("Show powerband shading", value=True)
    show_torque    = st.checkbox("Show wheel torque", value=True)
    st.divider()
    st.caption("Combo A — solid lines (blue palette)")
    st.caption("Combo B — dashed lines (red palette)")

# ---------------------------------------------------------------------------
# COMBO SELECTORS
# ---------------------------------------------------------------------------

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Combo A")
    gb_a_name    = st.selectbox("Gearbox", gearbox_names,
                                index=gearbox_names.index("ZF 8HP50 8-Speed"),
                                key="gb_a")
    diff_a_label = st.selectbox("Differential ratio", diff_labels,
                                index=diff_labels.index("4.375:1"), key="diff_a")
    skip_1st_a   = st.checkbox("Skip 1st gear (short gears only)",
                               value=(gb_a_name == "ZF 8HP50 8-Speed"), key="skip_a")

with col_b:
    st.markdown("### Combo B")
    gb_b_name    = st.selectbox("Gearbox", gearbox_names,
                                index=gearbox_names.index("SR20 5-Speed (FS5R90A)"),
                                key="gb_b")
    diff_b_label = st.selectbox("Differential ratio", diff_labels,
                                index=diff_labels.index("4.375:1"), key="diff_b")
    skip_1st_b   = st.checkbox("Skip 1st gear (short gears only)",
                               value=False, key="skip_b")

gb_a   = GEARBOXES[gb_a_name]
gb_b   = GEARBOXES[gb_b_name]
diff_a = DIFF_RATIOS[diff_a_label]
diff_b = DIFF_RATIOS[diff_b_label]
idx_a  = 1 if skip_1st_a else 0
idx_b  = 1 if skip_1st_b else 0

label_a = f"A-{gb_a['short']} ({diff_a_label})"
label_b = f"B-{gb_b['short']} ({diff_b_label})"

# ---------------------------------------------------------------------------
# BUILD CHART
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(16, 8))

title_main = (
    f"Sawtooth Shift Diagram\n"
    f"A: {gb_a_name}  |  diff {diff_a_label}   vs   "
    f"B: {gb_b_name}  |  diff {diff_b_label}"
)
fig.suptitle(title_main, fontsize=11, fontweight="bold", y=0.995)

subtitle = (
    f"Shift point: {shift_rpm:,} rpm  |  "
    f"Tyre: {TYRE_WIDTH_MM}/{int(TYRE_ASPECT_RATIO*100)}R{RIM_DIAMETER_INCH}  |  "
    f"Redline: {RPM_REDLINE:,} rpm"
)
ax.set_title(subtitle, fontsize=9, color="#555555", pad=6)

# Powerband shading
if show_powerband:
    ax.axhspan(RPM_POWERBAND_LOW, RPM_POWERBAND_HIGH,
               alpha=0.07, color="green", label="Powerband")

# --- Combo A (solid) ---
draw_sawtooth(ax, gb_a, diff_a, COLORS_A,
              linewidth=2.8, linestyle="-",
              combo_label=label_a, suffix="(A)",
              shift_rpm=shift_rpm, circ_m=circ_m,
              start_gear_idx=idx_a, origin_dotted=False)

# --- Combo B (dashed) ---
draw_sawtooth(ax, gb_b, diff_b, COLORS_B,
              linewidth=2.0, linestyle="--",
              combo_label=label_b, suffix="(B)",
              shift_rpm=shift_rpm, circ_m=circ_m,
              start_gear_idx=idx_b, origin_dotted=True)

# --- Wheel torque (secondary axis) ---
if show_torque:
    ax2 = ax.twinx()
    draw_torque(ax2, gb_a, diff_a, COLORS_A, shift_rpm=shift_rpm, circ_m=circ_m,
                start_gear_idx=idx_a, linewidth=2.0, alpha=0.85, fill_alpha=0.08)
    draw_torque(ax2, gb_b, diff_b, COLORS_B, shift_rpm=shift_rpm, circ_m=circ_m,
                start_gear_idx=idx_b, linewidth=1.4, alpha=0.60, fill_alpha=0.04)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Wheel Torque (Nm)  [dashed lines]",
                   fontsize=9, color="#993333", labelpad=10)
    ax2.tick_params(axis="y", labelcolor="#993333", labelsize=8)
    ax2.spines["right"].set_edgecolor("#993333")

# Shift and redline markers
ax.axhline(shift_rpm, color="orange", linestyle="--",
           linewidth=1.2, alpha=0.85, label=f"Shift point ({shift_rpm:,} rpm)")
ax.axhline(RPM_REDLINE, color="red", linestyle=":",
           linewidth=1.2, label=f"Redline ({RPM_REDLINE:,} rpm)")

# 100 km/h reference
ax.axvline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(101, RPM_REDLINE * 0.05, "100 km/h", fontsize=7.5, color="gray")

ax.set_xlabel("Road Speed (km/h)", fontsize=10)
ax.set_ylabel("Engine RPM", fontsize=10)
ax.set_xlim(0, x_max)
ax.set_ylim(0, RPM_REDLINE * 1.12)
ax.set_xticks(np.arange(0, x_max + 10, 10))
ax.tick_params(axis="x", labelrotation=45)
ax.grid(True, alpha=0.2)

# Deduplicated legend
handles, labels_list = ax.get_legend_handles_labels()
seen = {}
for h, lbl in zip(handles, labels_list):
    if lbl not in seen:
        seen[lbl] = h
ax.legend(seen.values(), seen.keys(),
          loc="upper right", fontsize=7.5, ncol=3, framealpha=0.88)

plt.tight_layout(rect=[0, 0, 1, 0.96])
st.pyplot(fig)
plt.close()

# ---------------------------------------------------------------------------
# STATS TABLE
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Summary stats")

stats_a = combo_stats(gb_a, diff_a, circ_m)
stats_b = combo_stats(gb_b, diff_b, circ_m)

col_sa, col_sb = st.columns(2)

with col_sa:
    st.markdown(f"**Combo A — {gb_a['short']}  diff {diff_a_label}**")
    for k, v in stats_a.items():
        st.markdown(f"- {k}: **{v}**")

with col_sb:
    st.markdown(f"**Combo B — {gb_b['short']}  diff {diff_b_label}**")
    for k, v in stats_b.items():
        st.markdown(f"- {k}: **{v}**")

# Per-gear speed table
st.divider()
with st.expander("Per-gear speed table (km/h at key RPMs)"):
    rpm_cols = [2000, 3000, 4000, 5000, 6000, 7000, RPM_REDLINE]

    def speed_row(gb, diff, start_idx):
        rows = {}
        ratios = list(gb["ratios"].values())
        gears  = list(gb["ratios"].keys())
        for i in range(start_idx, len(ratios)):
            gear = gears[i]
            row = {f"{r:,} rpm": f"{speed_kmh(r, ratios[i], diff, circ_m):.0f}" for r in rpm_cols}
            row["Overall ratio"] = f"{ratios[i] * diff:.3f}:1"
            rows[gear] = row
        return rows

    import pandas as pd
    rows_a = speed_row(gb_a, diff_a, 0)
    rows_b = speed_row(gb_b, diff_b, 0)

    df_a = pd.DataFrame(rows_a).T
    df_b = pd.DataFrame(rows_b).T

    st.markdown(f"**Combo A — {gb_a['short']}  diff {diff_a_label}**")
    st.dataframe(df_a, use_container_width=True)

    st.markdown(f"**Combo B — {gb_b['short']}  diff {diff_b_label}**")
    st.dataframe(df_b, use_container_width=True)
