# Drum-boiler dynamics — Åström & Bell (2000), Python implementation

Phase 1: 4th-order natural-circulation drum-boiler model with the paper's empirical closure relations.

## Files
| file | purpose |
|---|---|
| `model.py` | Eq. 24/25 state model, closures (Eqs. 12, 15, 18), steady state (Eq. 27), outputs (Eqs. 8, 19-22) |
| `steam.py`, `steam_coeffs.py`, `generate_steam_fit.py` | polynomial fits of saturated steam tables (IAPWS-IF97, 6-12 MPa) |
| `scenarios.py` | step / PRBS input builders, medium & high load operating points |
| `plots.py`, `app.py` | Plotly panels and the Streamlit dashboard (`streamlit run app.py`) |
| `make_paper_figures.py` | regenerates analogues of paper Figs 4-7 into `figures/` |
| `tests/` | steady state, Eq. 26 derivatives, global mass/energy balance, shrink-and-swell sign |

## Assumptions NOT in the paper (must be confirmed / replaced with Thermax data)
* `Adc` (downcomer area) = 0.35 m² — Eq. 15 needs it but only Vdc is listed. Chosen so medium load gives αr ≈ 0.053, qdc ≈ 1160 kg/s (paper Fig. 4: ≈0.052, ≈1200).
* `md` (drum metal mass, Eq. 17/25) = 0 — not among the nine parameters; 0 best reproduces the Fig. 6 level components.
* `Cp` (metal) = 500 J/kg/K; `Vwd_nom` = 20 m³ (half-full drum); `Vsd_nom` = 5 m³ at medium load, which fixes `V0sd` (Eq. 18).
* Medium load: 8.5 MPa, 50 kg/s steam (Q ≈ 87.5 MW); High load: 10.5 MPa, 100 kg/s; feedwater enthalpy 1000 kJ/kg. Read off the paper's figure axes.
* Cubic (not quadratic) steam-table fits: <0.02 % error vs IAPWS-97.

## Validation status
* Step responses (Figs 4-7): same states, shapes and signs; shrink/swell and medium-vs-high-load differences reproduced. Magnitudes are close but not identical (e.g. pressure ramp ≈ 2.8 kPa/s vs ≈ 2 kPa/s read from Fig. 4).
* Plant comparisons (Figs 8-16) need the Lund/Sydkraft PRBS data, which the paper does not tabulate. Use the dashboard's "Plant data (CSV)" mode (t, Q_MW, qf, qs [, p_MPa, level_m]).
