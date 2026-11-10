"""
Astrom & Bell (2000), "Drum-boiler dynamics", Automatica 36, 363-378.
Fourth-order nonlinear model of a natural-circulation drum boiler.

States  x = [Vwt, p, alpha_r, Vsd]
        Vwt  total water volume in system [m3]
        p    drum pressure [Pa]
        ar   steam mass fraction at riser outlet [-]
        Vsd  steam volume under the drum liquid level [m3]
Inputs  Q  heat flow to risers [W],  qf feedwater flow [kg/s],  qs steam flow [kg/s],
        hf feedwater specific enthalpy [J/kg]

Equation numbers in comments refer to the paper.
The EMPIRICAL closure relations (phase 1) are isolated in the `closure_*` methods and
in `_void_*` so they can be replaced by physics-based ones in phase 2.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import steam

G = 9.81


@dataclass
class Params:
    # Swedish P16-G16 plant values quoted in Sec. 5.1
    Vd: float = 40.0        # drum volume [m3]
    Vr: float = 37.0        # riser volume [m3]
    Vdc: float = 11.0       # downcomer volume [m3]
    Ad: float = 20.0        # drum wet surface area at operating level [m2]
    mt: float = 300_000.0   # total metal mass [kg]
    mr: float = 160_000.0   # riser metal mass [kg]
    k: float = 25.0         # friction coeff, downcomer-riser loop [-]
    beta: float = 0.3       # empirical parameter in Eq. 18 [-]
    Td: float = 12.0        # steam residence time in drum [s]
    # ---- NOT given in the paper's parameter list; assumptions (see README) ----
    Adc: float = 0.35       # downcomer cross-section [m2]  (calibrated: alpha_r ~ 0.052 at medium load)
    md: float = 0.0         # drum metal mass in Eq. 17/25 [kg]; paper lists none -> 0 reproduces Figs 4-7 best
    Cp: float = 500.0       # metal specific heat [J/kg/K]
    Vwd_nom: float = 20.0   # nominal water volume in drum (half-full) [m3]
    Vsd_nom: float = 5.0    # steam volume under level at the medium-load reference point [m3]
    V0sd: float | None = None   # Eq. 18 reference steam volume; set by calibrate_V0sd()

    @property
    def Vt(self):
        return self.Vd + self.Vr + self.Vdc


# ----------------------------------------------------------------------------------
# Void-fraction closure (Eqs. 10-12, 26): linear steam quality along riser
# ----------------------------------------------------------------------------------
def void_fraction(ar, rs, rw):
    """Average steam volume fraction in risers, Eq. 12."""
    ar = np.maximum(ar, 1e-7)
    D = rw - rs
    eta = ar * D / rs
    return rw / D * (1.0 - np.log1p(eta) / eta)


def void_derivs(ar, rs, rw, drs, drw):
    """d(abar_v)/d(alpha_r) and d(abar_v)/dp  (Eq. 26, differentiated analytically)."""
    ar = np.maximum(ar, 1e-7)
    D = rw - rs
    eta = ar * D / rs
    L = np.log1p(eta)
    # d/d(alpha_r)
    dav_dar = rw / rs * (L / eta**2 - 1.0 / (eta * (1.0 + eta)))
    # d/dp via chain rule on g(rw, D, eta)
    A = 1.0 - L / eta
    dg_drw_direct = A / D
    dg_dD = -rw / D**2 * A
    dg_deta = rw / D * (-1.0 / ((1.0 + eta) * eta) + L / eta**2)
    deta_drw = ar / rs
    deta_drs = -ar * rw / rs**2
    dav_drw = dg_drw_direct + dg_dD + dg_deta * deta_drw
    dav_drs = -dg_dD + dg_deta * deta_drs
    dav_dp = dav_drw * drw + dav_drs * drs
    return dav_dar, dav_dp


class DrumBoiler:
    def __init__(self, params: Params | None = None):
        self.P = params or Params()
        self.p_ref = None       # if set, steam flow follows pressure: qs_eff = qs_cmd * p / p_ref
        if self.P.V0sd is None:
            self.calibrate_V0sd()

    # ---------------- closure relations (EMPIRICAL, phase 1) ----------------------
    def closure_downcomer_flow(self, av, rs, rw):
        """Steady-state momentum balance, Eq. 15."""
        P = self.P
        return np.sqrt(2.0 * rw * P.Adc * (rw - rs) * G * av * P.Vr / P.k)

    def closure_steam_through_surface(self, Vsd, ar, qdc, qr, rs):
        """Empirical model for steam flow through the drum liquid surface, Eq. 18."""
        P = self.P
        return rs / P.Td * (Vsd - P.V0sd) + ar * qdc + ar * P.beta * (qdc - qr)

    # ---------------------------- helpers ----------------------------------------
    def _coeffs(self, x):
        """Assemble matrix E (Eq. 24/25) and return helper quantities."""
        P = self.P
        Vwt, p, ar, Vsd = x
        s = steam.props(p)
        rs, rw, hs, hw, hc = s["rho_s"], s["rho_w"], s["h_s"], s["h_w"], s["h_c"]
        drs, drw, dhs, dhw, dts = s["drho_s"], s["drho_w"], s["dh_s"], s["dh_w"], s["dt_s"]
        Vst = P.Vt - Vwt
        av = void_fraction(ar, rs, rw)
        dav_dar, dav_dp = void_derivs(ar, rs, rw, drs, drw)
        Vwd = Vwt - P.Vdc - (1.0 - av) * P.Vr                      # Eq. 19

        e11 = rw - rs
        e12 = Vwt * drw + Vst * drs
        e21 = rw * hw - rs * hs
        e22 = (Vwt * (hw * drw + rw * dhw) + Vst * (hs * drs + rs * dhs)
               - P.Vt + P.mt * P.Cp * dts)
        e32 = ((rw * dhw - ar * hc * drw) * (1 - av) * P.Vr
               + ((1 - ar) * hc * drs + rs * dhs) * av * P.Vr
               + (rs + (rw - rs) * ar) * hc * P.Vr * dav_dp
               - P.Vr + P.mr * P.Cp * dts)
        e33 = ((1 - ar) * rs + ar * rw) * hc * P.Vr * dav_dar
        e42 = (Vsd * drs
               + (rs * Vsd * dhs + rw * Vwd * dhw - (Vsd + Vwd) + P.md * P.Cp * dts) / hc
               + ar * (1 + P.beta) * P.Vr * (av * drs + (1 - av) * drw + (rs - rw) * dav_dp))
        e43 = ar * (1 + P.beta) * (rs - rw) * P.Vr * dav_dar
        e44 = rs
        E = np.array([[e11, e12, 0, 0],
                      [e21, e22, 0, 0],
                      [0, e32, e33, 0],
                      [0, e42, e43, e44]])
        return E, dict(s=s, av=av, Vwd=Vwd, Vst=Vst)

    # ---------------------------- dynamics ---------------------------------------
    def rhs(self, t, x, u):
        """u(t) -> (Q, qf, qs, hf). Returns dx/dt (Eq. 24)."""
        P = self.P
        Q, qf, qs, hf = u(t)
        if self.p_ref:
            qs = qs * x[1] / self.p_ref
        E, h = self._coeffs(x)
        s, av = h["s"], h["av"]
        Vwt, p, ar, Vsd = x
        rs, rw, hs, hw, hc = s["rho_s"], s["rho_w"], s["h_s"], s["h_w"], s["h_c"]
        qdc = self.closure_downcomer_flow(av, rs, rw)
        b = np.array([qf - qs,
                      Q + qf * hf - qs * hs,
                      Q - ar * hc * qdc,
                      rs / P.Td * (P.V0sd - Vsd) + (hf - hw) / hc * qf])
        return np.linalg.solve(E, b)

    def outputs(self, x, xdot, u_vals):
        """Algebraic/derived variables at one point (Sec. 4.5, Eqs. 8, 19, 20, 22)."""
        P = self.P
        Q, qf, qs, hf = u_vals
        if self.p_ref:
            qs = qs * x[1] / self.p_ref
        E, h = self._coeffs(x)
        s, av, Vwd, Vst = h["s"], h["av"], h["Vwd"], h["Vst"]
        Vwt, p, ar, Vsd = x
        rs, rw, hs, hw, hc = s["rho_s"], s["rho_w"], s["h_s"], s["h_w"], s["h_c"]
        dpdt, dardt = xdot[1], xdot[2]
        dav_dar, dav_dp = void_derivs(ar, rs, rw, s["drho_s"], s["drho_w"])
        qdc = self.closure_downcomer_flow(av, rs, rw)
        # Eq. 22
        qr = qdc - P.Vr * ((av * s["drho_s"] + (1 - av) * s["drho_w"] + (rs - rw) * dav_dp) * dpdt
                           + (rs - rw) * dav_dar * dardt)
        # Eq. 8 (total condensation flow)
        qct = ((hw - hf) / hc * qf
               + (rs * Vst * s["dh_s"] + rw * Vwt * s["dh_w"] - P.Vt
                  + P.mt * P.Cp * s["dt_s"]) / hc * dpdt)
        qsd = self.closure_steam_through_surface(Vsd, ar, qdc, qr, rs)
        return dict(qs=qs, qf=qf, Q=Q, p=p, Vwt=Vwt, ar=ar, Vsd=Vsd, av=av, Vwd=Vwd, qdc=qdc, qr=qr, qct=qct,
                    qsd=qsd, ts=s["t_s"], hc=hc)

    # ---------------------------- initial condition ------------------------------
    def steady_state(self, p, qs, hf, Vwd=None):
        """Equilibrium (Sec. 4.8): give p, qs, hf -> returns x0, u0."""
        P = self.P
        s = steam.props(p)
        rs, rw, hs, hw, hc = s["rho_s"], s["rho_w"], s["h_s"], s["h_w"], s["h_c"]
        qf = qs
        Q = qs * hs - qf * hf

        def f(ar):
            av = void_fraction(ar, rs, rw)
            return ar * hc * self.closure_downcomer_flow(av, rs, rw) - Q
        ar = brentq(f, 1e-6, 0.9)                                    # Eq. 27
        av = void_fraction(ar, rs, rw)
        Vsd = P.V0sd - P.Td * (hw - hf) / (rs * hc) * qf
        Vwd = P.Vwd_nom if Vwd is None else Vwd
        Vwt = Vwd + P.Vdc + (1 - av) * P.Vr
        return np.array([Vwt, p, ar, Vsd]), (Q, qf, qs, hf)

    def calibrate_V0sd(self, p=8.5e6, qs=50.0, hf=1.0e6):
        """Choose V0sd so that Vsd = Vsd_nom at the medium-load reference point."""
        s = steam.props(p)
        self.P.V0sd = self.P.Vsd_nom + self.P.Td * (s["h_w"] - hf) / (s["rho_s"] * s["h_c"]) * qs

    # ---------------------------- simulation -------------------------------------
    def simulate(self, x0, u, t_end, dt=1.0, rtol=1e-8, atol=1e-8, turbine_feedback=False):
        self.p_ref = x0[1] if turbine_feedback else None
        t_eval = np.arange(0, t_end + dt / 2, dt)
        sol = solve_ivp(self.rhs, (0, t_end), x0, args=(u,), method="LSODA",
                        t_eval=t_eval, rtol=rtol, atol=atol, max_step=min(2.0, dt))
        if not sol.success:
            raise RuntimeError(sol.message)
        res = {"t": sol.t}
        rows = []
        for i, t in enumerate(sol.t):
            xi = sol.y[:, i]
            xd = self.rhs(t, xi, u)
            rows.append(self.outputs(xi, xd, u(t)))
        for k in rows[0]:
            res[k] = np.array([r[k] for r in rows])
        # drum level deviation (Eq. 20), relative to initial values
        res["lw"] = (res["Vwd"] - res["Vwd"][0]) / self.P.Ad
        res["ls"] = (res["Vsd"] - res["Vsd"][0]) / self.P.Ad
        res["l"] = res["lw"] + res["ls"]
        return res
