"""Streamlit dashboard for the Astrom-Bell (2000) drum-boiler model.  Run: streamlit run app.py"""
from __future__ import annotations
import io
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model import DrumBoiler, Params
from scenarios import OPERATING_POINTS, step_inputs, prbs_inputs
import plots

st.set_page_config(page_title="Drum-boiler dynamics", layout="wide")
st.title("Drum-boiler dynamics — Åström & Bell Model")

sb = st.sidebar
sb.header("Operating point")
op_name = sb.selectbox("Preset", list(OPERATING_POINTS) + ["Custom"])
base = OPERATING_POINTS.get(op_name, OPERATING_POINTS["Medium load"])
p0 = sb.number_input("Drum pressure [MPa]", 6.5, 11.5, base["p"] / 1e6, 0.1)
qs0 = sb.number_input("Steam flow [kg/s]", 10.0, 200.0, base["qs"], 5.0)
hf0 = sb.number_input("Feedwater enthalpy [kJ/kg]", 400.0, 1400.0, base["hf"] / 1e3, 10.0)

with sb.expander("Plant parameters (Fixed + Assumptions)"):
    d = Params()
    pv = {}
    for name, val, hint in [
        ("Vd", d.Vd, "drum volume [m³]"), ("Vr", d.Vr, "riser volume [m³]"),
        ("Vdc", d.Vdc, "downcomer volume [m³]"), ("Ad", d.Ad, "drum wet area [m²]"),
        ("mt", d.mt, "total metal mass [kg]"), ("mr", d.mr, "riser metal mass [kg]"),
        ("k", d.k, "friction coeff"), ("beta", d.beta, "β in Eq. 18"), ("Td", d.Td, "steam residence time [s]"),
        ("Adc", d.Adc, "⚠ downcomer area [m²] (not in paper)"),
        ("md", d.md, "⚠ drum metal mass [kg] (not in paper)"),
        ("Cp", d.Cp, "⚠ metal Cp [J/(kg*K)] (not in paper)")]:
        pv[name] = st.number_input(f"{name} — {hint}", value=float(val), format="%g", key=name)

sb.header("Scenario")
mode = sb.radio("Excitation", ["Step", "PRBS (random binary)", "Plant data (CSV)"])
channel = sb.selectbox("Input channel", ["Q — fuel/heat flow [MW]", "qs — steam flow [kg/s]", "qf — feedwater [kg/s]"])
ch = channel.split(" ")[0]
default_amp = {"Q": 10.0, "qs": 10.0, "qf": 10.0}[ch]
amp = sb.number_input("Amplitude (MW or kg/s)", -100.0, 100.0, default_amp, 1.0)
t_step = sb.number_input("Step time [s]", 0.0, 1000.0, 20.0, 10.0)
t_end = sb.number_input("Duration [s]", 50.0, 4000.0, 200.0, 50.0)
tfb = sb.checkbox("Turbine feedback: steam flow ∝ pressure", value=(mode.startswith("PRBS"))) if mode != "Plant data (CSV)" else False
compare = sb.checkbox("Overlay medium vs high load (Figs 6/7 style)", value=False)

@st.cache_data(show_spinner="Integrating model…")
def run(pv_items, p_pa, qs, hf, ch, amp, t_step, t_end, mode, seed=1, data=None, tfb=False):
    params = Params(**dict(pv_items))
    m = DrumBoiler(params)                      
    x0, u0 = m.steady_state(p_pa, qs, hf)
    scale = 1e6 if ch == "Q" else 1.0
    if mode == "Step":
        u = step_inputs(u0, ch, amp * scale, t_step)
    elif mode.startswith("PRBS"):
        u = prbs_inputs(u0, ch, amp * scale, t_end, seed=seed)
    else:
        df = pd.DataFrame(data)
        t_d = df["t"].to_numpy(float)
        Qd = np.interp
        def u(t):
            return (np.interp(t, t_d, df["Q_MW"]) * 1e6, np.interp(t, t_d, df["qf"]),
                    np.interp(t, t_d, df["qs"]), hf)
        x0, _ = m.steady_state(p_pa, float(df["qs"].iloc[0]), hf)
        u0 = u(0.0)
    return m.simulate(x0, u, t_end, dt=1.0, turbine_feedback=tfb), u0

data = None
if mode == "Plant data (CSV)":
    up = sb.file_uploader("CSV columns: t [s], Q_MW, qf, qs  (optional: p_MPa, level_m)", type="csv")
    if up is None:
        st.info("Upload a plant-data CSV in the sidebar to simulate with measured inputs.")
        st.stop()
    data = pd.read_csv(up).to_dict("list")

pv_items = tuple(sorted(pv.items()))
runs, labels = [], []
try:
    if compare:
        for nm, o in OPERATING_POINTS.items():
            r, _ = run(pv_items, o["p"], o["qs"], o["hf"], ch, amp, t_step, t_end, mode, data=data, tfb=tfb)
            runs.append(r); labels.append(nm)
    else:
        r, u0 = run(pv_items, p0 * 1e6, qs0, hf0 * 1e3, ch, amp, t_step, t_end, mode, data=data, tfb=tfb)
        runs.append(r); labels.append(op_name)
except Exception as e:
    st.error(f"Simulation failed: {e}"); st.stop()

r0 = runs[0]
c = st.columns(5)
c[0].metric("Initial αr", f"{r0['ar'][0]:.4f}")
c[1].metric("Initial ᾱv", f"{r0['av'][0]:.3f}")
c[2].metric("Initial circulation qdc", f"{r0['qdc'][0]:.0f} kg/s")
c[3].metric("Peak level deviation", f"{np.max(np.abs(r0['l'])):.3f} m")
c[4].metric("Final pressure", f"{r0['p'][-1]/1e6:.2f} MPa")

tab1, tab2, tab3, tab4 = st.tabs(["Level dynamics", "State variables & flows", "Steady state (Fig. 3)", "Data / export"])
shift = t_step if mode == "Step" else 0.0
with tab1:
    fig = plots.level_panel(runs, labels)
    if data is not None and "level_m" in data:
        fig.add_trace(go.Scatter(x=data["t"], y=np.array(data["level_m"]) - data["level_m"][0], mode="markers",
                                 name="plant level", marker=dict(size=3, color="black")), row=1, col=1)
    if data is not None and "p_MPa" in data:
        fig.add_trace(go.Scatter(x=data["t"], y=data["p_MPa"], mode="markers", name="plant p",
                                 marker=dict(size=3, color="black")), row=1, col=2)
    st.plotly_chart(fig, width="stretch")
    if mode == "Step":
        st.caption("Shrink-and-swell: after a steam-flow step the level first moves opposite to the long-term water-balance "
                   "trend (non-minimum-phase). Compare medium vs high load with the overlay option.")
with tab2:
    st.plotly_chart(plots.state_panel(runs, labels), width="stretch")
with tab3:
    m = DrumBoiler(Params(**pv))
    pp = st.slider("Pressure [MPa]", 6.5, 11.5, float(p0), 0.5)
    st.plotly_chart(plots.fig3_curve(m, pp * 1e6), width="stretch")
    st.caption("A given change in input power produces a larger change in average steam-volume ratio at low power, "
               "which is why shrink/swell is stronger at low load (Eq. 27).")
with tab4:
    df = pd.DataFrame({k: v for k, v in runs[0].items()})
    st.dataframe(df.iloc[::5], width="stretch", height=300)
    st.download_button("Download full results (CSV)", df.to_csv(index=False), "drum_boiler_results.csv", "text/csv")
