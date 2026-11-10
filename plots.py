"""Plotly panel builders mirroring the layout of Figs 4-7 in the paper."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _panel(spec, results, labels, t_shift=0.0, height=760):
    """spec: list of (title, key-or-callable, scale, unit) laid out row-major in 2 columns."""
    rows = int(np.ceil(len(spec) / 2))
    fig = make_subplots(rows=rows, cols=2, subplot_titles=[s[0] for s in spec],
                        vertical_spacing=0.09, horizontal_spacing=0.08)
    for ri, res in enumerate(results):
        for i, (title, keys, scale, unit) in enumerate(spec):
            r, c = i // 2 + 1, i % 2 + 1
            keys = keys if isinstance(keys, (list, tuple)) else [keys]
            for kj, key in enumerate(keys):
                name = labels[ri] if len(keys) == 1 else f"{labels[ri]} {key}"
                dash = "solid" if kj == 0 else "dot"
                fig.add_trace(go.Scatter(x=res["t"] - t_shift, y=res[key] * scale, mode="lines",
                                         name=name, legendgroup=name, showlegend=(i == 0),
                                         line=dict(color=COLORS[ri % 4], dash=dash, width=2)), row=r, col=c)
            fig.update_yaxes(title_text=unit, row=r, col=c, title_font_size=11)
            if r == rows:
                fig.update_xaxes(title_text="time [s]", row=r, col=c)
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=1.08))
    return fig


def state_panel(results, labels, **kw):          # Figs 4 / 5
    spec = [("Drum pressure p", "p", 1e-6, "MPa"),
            ("Total water volume Vwt", "Vwt", 1, "m³"),
            ("Steam quality at riser outlet αr", "ar", 1, "–"),
            ("Steam volume in drum Vsd", "Vsd", 1, "m³"),
            ("Riser (qr) & downcomer (qdc) flow", ["qr", "qdc"], 1, "kg/s"),
            ("Total condensation flow qct", "qct", 1, "kg/s")]
    return _panel(spec, results, labels, **kw)


def level_panel(results, labels, **kw):          # Figs 6 / 7
    spec = [("Drum water level l", "l", 1, "m"),
            ("Drum pressure p", "p", 1e-6, "MPa"),
            ("Water-level contribution lw", "lw", 1, "m"),
            ("Steam volume ratio ᾱv", "av", 1, "–"),
            ("Steam-level contribution ls", "ls", 1, "m"),
            ("Steam quality αr", "ar", 1, "–")]
    return _panel(spec, results, labels, **kw)


def fig3_curve(model, p, n=60):
    """Steady-state ᾱv and αr vs input power (Fig. 3) at pressure p."""
    import steam
    from model import void_fraction
    s = steam.props(p)
    ars = np.linspace(1e-3, 0.2, n)
    avs = void_fraction(ars, s["rho_s"], s["rho_w"])
    Q = ars * s["h_c"] * model.closure_downcomer_flow(avs, s["rho_s"], s["rho_w"]) / 1e6
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Q, y=avs, name="steam volume ratio ᾱv", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=Q, y=ars, name="steam mass ratio αr", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Heat flow rate Q [MW]", yaxis_title="ratio [–]", height=420,
                      margin=dict(l=50, r=20, t=30, b=40))
    return fig
