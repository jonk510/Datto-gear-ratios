"""
Gearbox Sawtooth Comparison — Streamlit App
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gearbox Sawtooth Comparison", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
RPM_REDLINE    = 7500
RPM_PB_LOW     = 4500
RPM_PB_HIGH    = 7500
DRIVETRAIN_EFF = 0.95
START_RPM      = 1500

# Tyre 205/50R15
_sidewall  = 205 * 0.50
_diam_mm   = 15 * 25.4 + 2 * _sidewall
CIRC       = np.pi * _diam_mm / 1000   # metres

DIFF_RATIOS = {
    "3.540:1": 3.540, "3.700:1": 3.700, "3.900:1": 3.900,
    "4.110:1": 4.110, "4.375:1": 4.375,
}

GEARBOXES = {
    "ZF 8HP50 8-Speed": {
        "short": "ZF 8HP", "abbr": "8hp",
        "ratios": {"1st":4.714,"2nd":3.143,"3rd":2.106,"4th":1.667,
                   "5th":1.285,"6th":1.000,"7th":0.839,"8th":0.667},
    },
    "SR20 5-Speed (FS5R90A)": {
        "short": "SR20", "abbr": "SR",
        "ratios": {"1st":3.321,"2nd":2.077,"3rd":1.308,"4th":1.000,"5th":0.759},
    },
    "Nismo/HPI S15 6-Speed (FS6R31A)": {
        "short": "Nismo", "abbr": "nismo",
        "ratios": {"1st":3.083,"2nd":1.888,"3rd":1.330,"4th":1.034,"5th":0.834,"6th":0.667},
    },
}

TORQUE_MAP = {
    2000:195, 2500:205, 3000:215, 3500:240, 4000:285,
    4500:330, 5000:365, 5500:395, 6000:410, 6125:412.5,
    6500:405, 6790:390, 7000:370, 7200:350,
}
_TQ_RPMS = sorted(TORQUE_MAP.keys())
_TQ_VALS = [TORQUE_MAP[r] for r in _TQ_RPMS]

COLOR_A = "#1f77b4"
COLOR_B = "#d62728"

# ── Maths ──────────────────────────────────────────────────────────────────────
def speed_kmh(rpm, gear_ratio, diff_ratio):
    return (rpm / (gear_ratio * diff_ratio)) * CIRC * 60 / 1000

def rpm_at_speed(spd_kmh, gear_ratio, diff_ratio):
    return (spd_kmh * 1000 / (CIRC * 60)) * gear_ratio * diff_ratio

def engine_torque(rpm):
    return float(np.interp(rpm, _TQ_RPMS, _TQ_VALS))

def wheel_torque(rpm, gear_ratio, diff_ratio):
    return engine_torque(rpm) * gear_ratio * diff_ratio * DRIVETRAIN_EFF

def entry_rpm_after_skip(ratios, shift_rpm, start_idx):
    rpm = START_RPM
    for k in range(start_idx):
        rpm = shift_rpm * ratios[k + 1] / ratios[k]
    return rpm

# ── Trace builders ─────────────────────────────────────────────────────────────
def sawtooth_traces(gb, diff, color, dash, line_w, label, start_idx,
                     shift_rpm, show_labels):
    ratios = list(gb["ratios"].values())
    gears  = list(gb["ratios"].keys())
    origin_dash = "dot" if dash == "dash" else "solid"
    traces = []
    annotations = []

    entry_rpm = entry_rpm_after_skip(ratios, shift_rpm, start_idx)

    for i in range(start_idx, len(ratios)):
        gear  = gears[i]
        ratio = ratios[i]

        spd_entry = speed_kmh(entry_rpm, ratio, diff)
        spd_shift = speed_kmh(shift_rpm,  ratio, diff)

        # Faint origin extension
        traces.append(go.Scatter(
            x=[0, spd_entry], y=[0, entry_rpm],
            mode="lines",
            line=dict(color="#d0d0d0", width=0.9, dash=origin_dash),
            opacity=0.5, showlegend=False, hoverinfo="skip",
        ))

        # Rising diagonal
        traces.append(go.Scatter(
            x=[spd_entry, spd_shift], y=[entry_rpm, shift_rpm],
            mode="lines",
            line=dict(color=color, width=line_w, dash=dash),
            name=label, legendgroup=label,
            showlegend=(i == start_idx),
            hovertemplate=f"<b>{label} {gear}</b><br>%{{x:.0f}} km/h | %{{y:,.0f}} rpm<extra></extra>",
        ))

        # Gear label
        if show_labels:
            annotations.append(dict(
                x=(spd_entry + spd_shift) / 2,
                y=(entry_rpm  + shift_rpm)  / 2 + 200,
                xref="x", yref="y",
                text=f"<b>{gear}</b><br><span style='font-size:8px'>{gb['abbr']}</span>",
                showarrow=False,
                font=dict(size=8, color=color),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor=color, borderwidth=1, borderpad=3,
            ))

        # Vertical drop to next gear
        if i < len(ratios) - 1:
            next_entry = shift_rpm * ratios[i + 1] / ratio
            traces.append(go.Scatter(
                x=[spd_shift, spd_shift], y=[shift_rpm, next_entry],
                mode="lines",
                line=dict(color=color, width=line_w * 0.6, dash=dash),
                opacity=0.65, showlegend=False, hoverinfo="skip",
            ))
            entry_rpm = next_entry

    return traces, annotations


def torque_traces(gb, diff, color, start_idx, shift_rpm, line_w, alpha, fill_alpha):
    ratios = list(gb["ratios"].values())
    N = 120
    traces = []
    x_all, y_all = [], []

    entry_rpm = entry_rpm_after_skip(ratios, shift_rpm, start_idx)

    for i in range(start_idx, len(ratios)):
        ratio = ratios[i]
        rpm_pts = np.linspace(entry_rpm, shift_rpm, N)
        spd_pts = [speed_kmh(r, ratio, diff) for r in rpm_pts]
        wt_pts  = [wheel_torque(r, ratio, diff) for r in rpm_pts]

        # Filled polygon per gear
        x_poly = spd_pts + spd_pts[::-1]
        y_poly = wt_pts  + [0] * N
        r_int = int(color[1:3], 16)
        g_int = int(color[3:5], 16)
        b_int = int(color[5:7], 16)
        fill_col = f"rgba({r_int},{g_int},{b_int},{fill_alpha})"
        traces.append(go.Scatter(
            x=x_poly, y=y_poly,
            fill="toself", fillcolor=fill_col,
            mode="none",
            yaxis="y2", showlegend=False, hoverinfo="skip",
        ))

        x_all.extend(spd_pts)
        y_all.extend(wt_pts)

        if i < len(ratios) - 1:
            next_entry = shift_rpm * ratios[i + 1] / ratio
            x_all.append(spd_pts[-1])
            y_all.append(wheel_torque(next_entry, ratios[i + 1], diff))
            entry_rpm = next_entry

    # Continuous torque line
    traces.append(go.Scatter(
        x=x_all, y=y_all,
        mode="lines",
        line=dict(color=color, width=line_w, dash="dash"),
        opacity=alpha,
        yaxis="y2", showlegend=False,
        hovertemplate="%{y:.0f} Nm @ %{x:.0f} km/h<extra>Wheel Torque</extra>",
    ))
    return traces


# ── Stats ──────────────────────────────────────────────────────────────────────
def stats_rows(gb, diff):
    ratios = list(gb["ratios"].values())
    gears  = list(gb["ratios"].keys())
    n = len(ratios)
    top_spd = speed_kmh(RPM_REDLINE, ratios[-1], diff)
    rpm100  = rpm_at_speed(100, ratios[-1], diff)
    rpm110  = rpm_at_speed(110, ratios[-1], diff)
    return [
        ("Top gear",                    gears[-1]),
        ("Top speed",                   f"{top_spd:.1f} km/h  ({top_spd/1.609:.1f} mph)"),
        ("Cruise @ 100 km/h",           f"{rpm100:,.0f} rpm"),
        ("Cruise @ 110 km/h",           f"{rpm110:,.0f} rpm"),
        ("Overall 1st ratio",           f"{ratios[0]*diff:.3f}:1"),
        (f"Overall {gears[-1]} ratio",  f"{ratios[-1]*diff:.3f}:1"),
    ]


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Gearbox Sawtooth Comparison")
st.caption("Tyre: 205/50R15  |  Redline: 7,500 rpm  |  Drivetrain efficiency: 95%")

gb_names = list(GEARBOXES.keys())
diff_keys = list(DIFF_RATIOS.keys())

col_a, col_b, col_cfg = st.columns([2, 2, 1.4])

with col_a:
    st.markdown("**🔵 Combo A — solid lines**")
    gb_a_key  = st.selectbox("Gearbox", gb_names, index=0, key="gb_a")
    diff_a_key = st.selectbox("Differential ratio", diff_keys, index=4, key="diff_a")
    skip1_a   = st.checkbox("Skip 1st gear", value=(gb_a_key == "ZF 8HP50 8-Speed"), key="sk_a")

with col_b:
    st.markdown("**🔴 Combo B — dashed lines**")
    gb_b_key  = st.selectbox("Gearbox", gb_names, index=1, key="gb_b")
    diff_b_key = st.selectbox("Differential ratio", diff_keys, index=4, key="diff_b")
    skip1_b   = st.checkbox("Skip 1st gear", value=(gb_b_key == "ZF 8HP50 8-Speed"), key="sk_b")

with col_cfg:
    st.markdown("**Chart settings**")
    shift_rpm   = st.slider("Shift RPM", 5000, 7500, 7200, 100)
    x_max       = st.slider("Max speed (km/h)", 120, 320, 240, 10)
    show_pb     = st.checkbox("Powerband", value=True)
    show_torque = st.checkbox("Wheel Torque", value=True)
    show_labels = st.checkbox("Gear Labels", value=True)

# ── Build figure ───────────────────────────────────────────────────────────────
gb_a   = GEARBOXES[gb_a_key]
gb_b   = GEARBOXES[gb_b_key]
diff_a = DIFF_RATIOS[diff_a_key]
diff_b = DIFF_RATIOS[diff_b_key]
idx_a  = 1 if skip1_a else 0
idx_b  = 1 if skip1_b else 0

label_a = f"A · {gb_a['short']}"
label_b = f"B · {gb_b['short']}"

traces_a, ann_a = sawtooth_traces(gb_a, diff_a, COLOR_A, "solid", 2.8, label_a, idx_a, shift_rpm, show_labels)
traces_b, ann_b = sawtooth_traces(gb_b, diff_b, COLOR_B, "dash",  2.0, label_b, idx_b, shift_rpm, show_labels)

all_traces = traces_a + traces_b
annotations = ann_a + ann_b

if show_torque:
    all_traces += torque_traces(gb_a, diff_a, COLOR_A, idx_a, shift_rpm, 2.0, 0.85, 0.08)
    all_traces += torque_traces(gb_b, diff_b, COLOR_B, idx_b, shift_rpm, 1.4, 0.60, 0.04)

# Phantom legend entries
all_traces += [
    go.Scatter(x=[None], y=[None], mode="lines",
               line=dict(color="orange", dash="dash", width=1.5),
               name=f"Shift point ({shift_rpm:,} rpm)"),
    go.Scatter(x=[None], y=[None], mode="lines",
               line=dict(color="red", dash="dot", width=1.5),
               name=f"Redline ({RPM_REDLINE:,} rpm)"),
]
if show_torque:
    all_traces += [
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color=COLOR_A, dash="dash", width=2),
                   name="Wheel torque A (right axis)"),
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color=COLOR_B, dash="dash", width=2),
                   name="Wheel torque B (right axis)"),
    ]

# 100 km/h annotation
annotations.append(dict(
    x=101, y=RPM_REDLINE * 0.05, xref="x", yref="y",
    text="100 km/h", showarrow=False,
    font=dict(size=9, color="#888"), xanchor="left",
))

shapes = [
    dict(type="line", xref="paper", x0=0, x1=1, yref="y",
         y0=shift_rpm, y1=shift_rpm,
         line=dict(color="orange", width=1.2, dash="dash"), opacity=0.85),
    dict(type="line", xref="paper", x0=0, x1=1, yref="y",
         y0=RPM_REDLINE, y1=RPM_REDLINE,
         line=dict(color="red", width=1.2, dash="dot")),
    dict(type="line", xref="x", x0=100, x1=100, yref="paper", y0=0, y1=1,
         line=dict(color="#aaa", width=0.8, dash="dash"), opacity=0.6),
]
if show_pb:
    shapes.insert(0, dict(
        type="rect", xref="paper", x0=0, x1=1,
        yref="y", y0=RPM_PB_LOW, y1=RPM_PB_HIGH,
        fillcolor="rgba(0,160,0,0.07)", line=dict(width=0),
    ))

layout = go.Layout(
    title=dict(
        text=(f"<b>Sawtooth Shift Diagram</b><br>"
              f"<span style='font-size:11px;color:#555'>"
              f"A: {gb_a_key} | diff {diff_a_key}"
              f"&nbsp;&nbsp;vs&nbsp;&nbsp;"
              f"B: {gb_b_key} | diff {diff_b_key}</span>"),
        font=dict(size=15), y=0.99, yanchor="top",
    ),
    xaxis=dict(
        title="Road Speed (km/h)", range=[0, x_max], dtick=10,
        showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False,
    ),
    yaxis=dict(
        title="Engine RPM", range=[0, RPM_REDLINE * 1.12],
        showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False,
    ),
    yaxis2=dict(
        title="Wheel Torque (Nm)", overlaying="y", side="right",
        rangemode="tozero", showgrid=False, zeroline=False,
        tickfont=dict(color="#993333", size=9),
        title_font=dict(size=10, color="#993333"),
    ) if show_torque else None,
    annotations=annotations,
    shapes=shapes,
    legend=dict(
        x=0.99, y=0.99, xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc", borderwidth=1,
        font=dict(size=9), tracegroupgap=4,
    ),
    hovermode="closest",
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=80, r=80 if show_torque else 24, b=60, l=70),
    height=620,
)

fig = go.Figure(data=all_traces, layout=layout)
st.plotly_chart(fig, use_container_width=True)

# ── Stats ──────────────────────────────────────────────────────────────────────
st.divider()
sc_a, sc_b = st.columns(2)

with sc_a:
    st.markdown(f"**🔵 Combo A — {gb_a['short']} · diff {diff_a_key}**")
    st.dataframe(
        pd.DataFrame(stats_rows(gb_a, diff_a), columns=["", "Value"]).set_index(""),
        use_container_width=True)

with sc_b:
    st.markdown(f"**🔴 Combo B — {gb_b['short']} · diff {diff_b_key}**")
    st.dataframe(
        pd.DataFrame(stats_rows(gb_b, diff_b), columns=["", "Value"]).set_index(""),
        use_container_width=True)

st.caption("Double-click chart to reset zoom  |  Hover for values  |  Drag to zoom")
