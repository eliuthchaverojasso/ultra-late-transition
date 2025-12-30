#!/usr/bin/env python3
# =============================================================================
# Authors: Eliuth Chavero Jasso & Alexa Perez Cornejo
# =============================================================================
#
# INPUT FILES expected:
#   Pantheon+SH0ES.dat
#   Pantheon+SH0ES_STAT+SYS.cov
#   bao_data/desi_gaussian_bao_ALL_GCcomb_mean.txt
#   bao_data/desi_gaussian_bao_ALL_GCcomb_cov.txt
#
# USAGE:
#   python late_journal_grade_fixed.py --mode run --zcuts 1.5 1.2 1.0
#   python late_journal_grade_fixed.py --mode plot
#   python late_journal_grade_fixed.py --mode run --do_mcmc --mcmc_cases ALL SN+CC --mcmc_steps 60000
# =============================================================================

import os
import json
import time
import platform
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List, Callable
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from scipy.stats import chi2 as chi2_dist


# =============================================================================
# 0) CONFIG / LOGGING / STYLE
# =============================================================================

@dataclass(frozen=True)
class Config:
    OUTDIR: str = "paper_figures_PRD_JOURNAL"
    SEED: int = 12345

    # Cosmology constants
    c_light: float = 299792.458
    Om_m0: float = 0.315
    Om_r0: float = 9e-5

    # Pantheon nuisance prior
    M0: float = -19.253
    sigma_M: float = 0.027

    # Fig1/2 scan convention
    H0_PHYS_SCAN: float = 73.0
    FIG12_ZMAX: float = 2.5
    FIG12_N: int = 2500

    # Robust bounds
    H0_BOUNDS: Tuple[float, float] = (55.0, 85.0)
    RD_BOUNDS: Tuple[float, float] = (100.0, 180.0)

    # IMPORTANT: widened to match your profiler where d_late>0 can be preferred
    DLATE_BOUNDS: Tuple[float, float] = (-3.0, 3.0)
    ZT_BOUNDS: Tuple[float, float] = (0.001, 1.5)

    # 2D profile grid (log sampling in zt)
    N_GRID: int = 30
    DLATE_GRID_MIN: float = -1.2
    DLATE_GRID_MAX: float = 0.8
    ZT_GRID_MIN: float = 0.001
    ZT_GRID_MAX: float = 0.8

    # Background sampling
    N_BG: int = 4500
    ZMAX_MARGIN: float = 0.20
    ZMAX_CAP: float = 3.0

    # Robustness suites
    WIDTH_LIST: Tuple[float, ...] = (0.05, 0.10, 0.20)
    MODEL_LIST: Tuple[str, ...] = ("sigmoid", "tanh", "piecewise")

    # Optimization controls
    N_RESTARTS_GLOBAL: int = 16
    N_RESTARTS_H0RD: int = 10
    MIN_RESTARTS_WARM: int = 5

    # MCMC defaults
    MCMC_STEPS: int = 60000
    MCMC_BURN: int = 12000
    MCMC_THIN: int = 10

    # Cache safety
    BG_CACHE_MAXSIZE: int = 6000   # prevents multi-GB blowups
    ROUND_KEYS: int = 6            # rounding improves cache hits

    # Interp tolerance
    INTERP_EPS: float = 1e-8


CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("late_JOURNAL")
os.makedirs(CFG.OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.5,
    "axes.linewidth": 1.0,
    "savefig.bbox": "tight",
    "figure.dpi": 150
})

MODEL_SEED_TAG = {"sigmoid": 11, "tanh": 22, "piecewise": 33}  # [FIX-1]


# =============================================================================
# 1) UTILITIES
# =============================================================================

def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def save_figure(fig, filename: str) -> None:
    base = os.path.join(CFG.OUTDIR, filename)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)

def safe_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, name: str) -> np.ndarray:
    """Fail-fast interpolation with robust epsilon bounds check."""
    x = np.asarray(x, float)
    eps = float(CFG.INTERP_EPS)
    if x.size == 0:
        return np.asarray([], float)
    if (x.min() < xp.min() - eps) or (x.max() > xp.max() + eps):
        raise RuntimeError(
            f"{name}: requested x-range [{x.min():.6g},{x.max():.6g}] outside grid "
            f"[{xp.min():.6g},{xp.max():.6g}] (eps={eps}). Increase zmax or cut data."
        )
    return np.interp(x, xp, fp)

def infer_zmax(sn: Dict, desi: Dict, cc: Dict) -> float:
    z_sn = float(np.max(sn["z"])) if len(sn["z"]) else 0.0
    z_bao = float(np.max(desi["z"])) if len(desi["z"]) else 0.0
    z_cc = float(np.max(cc["z"])) if len(cc["z"]) else 0.0
    zmax = max(z_sn, z_bao, z_cc) + float(CFG.ZMAX_MARGIN) + 1e-6
    return min(zmax, float(CFG.ZMAX_CAP))

def aic(chi2: float, k: int) -> float:
    return float(chi2 + 2.0 * k)

def bic(chi2: float, k: int, n: int) -> float:
    return float(chi2 + k * np.log(max(int(n), 1)))

def lrt_pvalue_heuristic(delta_chi2: float, dof: int) -> float:
    """Heuristic Wilks p-value. NOTE: non-regular nesting applies here."""
    if delta_chi2 <= 0:
        return 1.0
    return float(1.0 - chi2_dist.cdf(delta_chi2, df=int(dof)))

def n_points(use_sn: bool, use_bao: bool, use_cc: bool, sn: Dict, desi: Dict, cc: Dict) -> int:
    return int((len(sn["z"]) if use_sn else 0) +
               (len(desi["z"]) if use_bao else 0) +
               (len(cc["z"]) if use_cc else 0))

def k_params_late(use_bao: bool) -> int:
    # d_late, zt, H0, (rd if BAO)
    return 4 if use_bao else 3

def k_params_lcdm(use_bao: bool) -> int:
    # H0, (rd if BAO)
    return 2 if use_bao else 1


# =============================================================================
# 2) DATA LOADING (zcut applies consistently to SN/BAO/CC)
# =============================================================================

def load_cc_df() -> pd.DataFrame:
    data = np.array([
        [0.070,  69.0, 19.6], [0.090,  69.0, 12.0], [0.120,  68.6, 26.2],
        [0.170,  83.0,  8.0], [0.179,  75.0,  4.0], [0.199,  75.0,  5.0],
        [0.200,  72.9, 29.6], [0.270,  77.0, 14.0], [0.280,  88.8, 36.6],
        [0.352,  83.0, 14.0], [0.380,  83.0, 13.5], [0.400,  95.0, 17.0],
        [0.4004, 77.0, 10.2], [0.4247, 87.1, 11.2], [0.4497, 92.8, 12.9],
        [0.470,  89.0, 50.0], [0.4783, 80.9,  9.0], [0.480,  97.0, 62.0],
        [0.593, 104.0, 13.0], [0.680,  92.0,  8.0], [0.781, 105.0, 12.0],
        [0.875, 125.0, 17.0], [0.880,  90.0, 40.0], [0.900, 117.0, 23.0],
        [1.037, 154.0, 20.0], [1.300, 168.0, 17.0], [1.363, 160.0, 33.6],
        [1.430, 177.0, 18.0], [1.530, 140.0, 14.0], [1.750, 202.0, 40.0],
        [1.965, 186.5, 50.4],
    ], dtype=float)
    return pd.DataFrame(data, columns=["z", "Hz", "sig"])

def load_data(zcut: float) -> Tuple[Dict, Dict, Dict]:
    # --- Pantheon+ ---
    df = pd.read_csv("Pantheon+SH0ES.dat", sep=r"\s+")
    z = df["zHD"].values.astype(float)
    mB = df["mB"].values.astype(float)
    x1 = df["x1"].values.astype(float)
    c = df["c"].values.astype(float)

    mask = z < float(zcut)
    z = z[mask]; mB = mB[mask]; x1 = x1[mask]; c = c[mask]

    with open("Pantheon+SH0ES_STAT+SYS.cov", "r") as f:
        N = int(f.readline().strip())
        raw = np.fromfile(f, sep=" ", dtype=float)
    C_full = raw.reshape((N, N))
    idx = np.where(mask)[0]
    C = C_full[np.ix_(idx, idx)]
    Cinv = np.linalg.inv(C)

    ones = np.ones_like(z)
    J = np.vstack([ones, -x1, c]).T
    JCinv = J.T @ Cinv
    P = np.linalg.inv(JCinv @ J) @ JCinv

    sn = {
        "z": z, "mB": mB, "C": C, "Cinv": Cinv, "P": P,
        "glob": (ones, -x1, c),
        "M0": CFG.M0, "sigma_M": CFG.sigma_M
    }
    if len(sn["z"]) < 50:
        raise RuntimeError(f"SN after zcut={zcut} too small: N={len(sn['z'])}")

    # --- DESI BAO ---
    mean_f = os.path.join("bao_data", "desi_gaussian_bao_ALL_GCcomb_mean.txt")
    cov_f  = os.path.join("bao_data", "desi_gaussian_bao_ALL_GCcomb_cov.txt")

    df_d = pd.read_csv(mean_f, sep=r"\s+", comment="#", header=None, names=["z", "val", "type"])
    mask_b = df_d["z"].values.astype(float) < float(zcut)
    idx_b = np.where(mask_b)[0]
    z_b = df_d.iloc[idx_b]["z"].values.astype(float)
    y_b = df_d.iloc[idx_b]["val"].values.astype(float)
    types = df_d.iloc[idx_b]["type"].astype(str).values

    cov_full = np.loadtxt(cov_f)
    cov = cov_full[np.ix_(idx_b, idx_b)]
    Cinv_b = np.linalg.inv(cov) if len(idx_b) else np.zeros((0, 0), float)

    tm = np.zeros(len(types), dtype=int)
    for i, t in enumerate(types):
        if "DV" in t: tm[i] = 0
        elif "DM" in t: tm[i] = 1
        elif "DH" in t: tm[i] = 2
        else:
            raise RuntimeError(f"Unknown BAO type string: {t}")

    desi = {"z": z_b, "y": y_b, "tm": tm, "C": cov, "Cinv": Cinv_b}

    # --- CC ---
    cc_df = load_cc_df()
    cc_df = cc_df[cc_df["z"] < float(zcut)].copy()
    cc = {"z": cc_df["z"].values.astype(float),
          "Hz": cc_df["Hz"].values.astype(float),
          "sig": cc_df["sig"].values.astype(float)}
    return sn, desi, cc


# =============================================================================
# 3) THEORY: models + cached background E(z), I(z)
# =============================================================================

def delta_sigmoid(z: np.ndarray, d_late: float, zt: float, w: float) -> np.ndarray:
    return d_late / (1.0 + np.exp((z - zt) / w))

def delta_tanh(z: np.ndarray, d_late: float, zt: float, w: float) -> np.ndarray:
    return 0.5 * d_late * (1.0 - np.tanh((z - zt) / w))

def delta_piecewise(z: np.ndarray, d_late: float, zt: float, w: float) -> np.ndarray:
    lo, hi = zt - w, zt + w
    out = np.zeros_like(z, dtype=float)
    out[z <= lo] = d_late
    out[z >= hi] = 0.0
    mid = (z > lo) & (z < hi)
    out[mid] = d_late * (hi - z[mid]) / max(hi - lo, 1e-15)
    return out

def _round_key(x: float) -> float:
    return float(np.round(float(x), CFG.ROUND_KEYS))

def _cache_key(model: str, d_late: float, zt: float, w: float, zmax: float) -> Tuple:
    return (model, _round_key(d_late), _round_key(zt), _round_key(w), _round_key(zmax))

@lru_cache(maxsize=CFG.BG_CACHE_MAXSIZE)  # [FIX-2]
def _cached_background(key: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, d_late_r, zt_r, w_r, zmax_r = key
    z = np.linspace(0.0, float(zmax_r), int(CFG.N_BG))
    if model == "sigmoid":
        d = delta_sigmoid(z, float(d_late_r), float(zt_r), float(w_r))
    elif model == "tanh":
        d = delta_tanh(z, float(d_late_r), float(zt_r), float(w_r))
    elif model == "piecewise":
        d = delta_piecewise(z, float(d_late_r), float(zt_r), float(w_r))
    else:
        raise ValueError(f"Unknown model: {model}")

    f = d / (1.0 + z)
    dz = z[1] - z[0]
    integ = np.zeros_like(z)
    integ[1:] = np.cumsum(0.5 * (f[1:] + f[:-1])) * dz
    rho_de = np.exp(3.0 * integ)

    O_de = 1.0 - float(CFG.Om_m0) - float(CFG.Om_r0)
    Esq = float(CFG.Om_m0) * (1 + z)**3 + float(CFG.Om_r0) * (1 + z)**4 + O_de * rho_de
    E = np.sqrt(np.maximum(Esq, 1e-14))

    invE = 1.0 / E
    I = np.zeros_like(z)
    I[1:] = np.cumsum(0.5 * (invE[1:] + invE[:-1])) * dz
    return z, I, E

def background(model: str, d_late: float, zt: float, w: float, zmax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cached_background(_cache_key(model, d_late, zt, w, zmax))


# =============================================================================
# 4) LIKELIHOODS
# =============================================================================

def chi2_sn(sn: Dict, z_g: np.ndarray, I_g: np.ndarray, H0: float) -> float:
    z = sn["z"]
    I = safe_interp(z, z_g, I_g, "SN I(z)")
    dl = np.maximum((1.0 + z) * (CFG.c_light / float(H0)) * I, 1e-30)
    mu = 5.0 * np.log10(dl) + 25.0
    dvec = mu - sn["mB"]
    nuis = -(sn["P"] @ dvec)
    g = sn["glob"]
    res = dvec + (nuis[0]*g[0] + nuis[1]*g[1] + nuis[2]*g[2])
    chi2 = float(res @ sn["Cinv"] @ res)
    prior = float(((nuis[0] - sn["M0"]) / sn["sigma_M"])**2)
    return chi2 + prior

def chi2_bao(desi: Dict, z_g: np.ndarray, I_g: np.ndarray, E_g: np.ndarray, H0: float, rd: float) -> float:
    z = desi["z"]
    if len(z) == 0:
        return 0.0
    I = safe_interp(z, z_g, I_g, "BAO I(z)")
    E = safe_interp(z, z_g, E_g, "BAO E(z)")
    DM = (CFG.c_light / float(H0)) * I
    DH = CFG.c_light / (float(H0) * E)
    DV = (z * DH * DM**2)**(1.0/3.0)
    tm = desi["tm"]
    model = np.zeros_like(z)
    model[tm == 0] = DV[tm == 0]
    model[tm == 1] = DM[tm == 1]
    model[tm == 2] = DH[tm == 2]
    res = (model / float(rd)) - desi["y"]
    return float(res @ desi["Cinv"] @ res)

def chi2_cc(cc: Dict, z_g: np.ndarray, E_g: np.ndarray, H0: float) -> float:
    z = cc["z"]
    if len(z) == 0:
        return 0.0
    Ez = safe_interp(z, z_g, E_g, "CC E(z)")
    pred = float(H0) * Ez
    return float(np.sum(((pred - cc["Hz"]) / cc["sig"])**2))


# =============================================================================
# 5) OPTIMIZATION: deterministic multistart + fallback
# =============================================================================

@dataclass
class FitResult:
    success: bool
    fun: float
    x: np.ndarray
    nit: int
    status: int
    message: str
    method: str
    n_success_starts: int = 0

def _pack(r, method: str, n_success: int = 0) -> FitResult:
    return FitResult(
        success=bool(r.success),
        fun=float(r.fun),
        x=np.array(r.x, float),
        nit=int(getattr(r, "nit", -1)),
        status=int(getattr(r, "status", -1)),
        message=str(getattr(r, "message", "")),
        method=method,
        n_success_starts=int(n_success),
    )

def minimize_multistart(fun: Callable[[np.ndarray], float],
                        x0: np.ndarray,
                        bounds: List[Tuple[float, float]],
                        rng: np.random.Generator,
                        n_starts: int,
                        jitter: np.ndarray) -> FitResult:
    best: Optional[FitResult] = None
    x0 = np.array(x0, float)
    n_success = 0

    for _ in range(int(n_starts)):
        x = x0 + rng.normal(0.0, jitter, size=x0.size)
        for i, (lo, hi) in enumerate(bounds):
            x[i] = float(np.clip(x[i], lo, hi))

        r = minimize(fun, x, bounds=bounds, method="L-BFGS-B")
        if bool(r.success):
            n_success += 1
        cand = _pack(r, "L-BFGS-B", n_success=n_success)
        if (best is None) or (cand.fun < best.fun):
            best = cand

    if best is None or (not best.success):
        r = minimize(fun, x0, bounds=bounds, method="Powell")
        best = _pack(r, "Powell", n_success=n_success)
    return best


# =============================================================================
# 6) GLOBAL LOSSES: late vs LCDM
# =============================================================================

def make_late_loss(model: str, width: float,
                    sn: Dict, desi: Dict, cc: Dict, zmax: float,
                    use_sn: bool, use_bao: bool, use_cc: bool) -> Callable[[np.ndarray], float]:
    def loss(theta: np.ndarray) -> float:
        d_late, zt, H0, rd = [float(v) for v in theta]

        if not (CFG.DLATE_BOUNDS[0] <= d_late <= CFG.DLATE_BOUNDS[1]): return 1e30
        if not (CFG.ZT_BOUNDS[0] <= zt <= CFG.ZT_BOUNDS[1]): return 1e30
        if not (CFG.H0_BOUNDS[0] <= H0 <= CFG.H0_BOUNDS[1]): return 1e30
        if use_bao and (not (CFG.RD_BOUNDS[0] <= rd <= CFG.RD_BOUNDS[1])): return 1e30

        z_g, I_g, E_g = background(model, d_late, zt, width, zmax)

        chi2 = 0.0
        if use_sn:  chi2 += chi2_sn(sn, z_g, I_g, H0)
        if use_bao: chi2 += chi2_bao(desi, z_g, I_g, E_g, H0, rd)
        if use_cc:  chi2 += chi2_cc(cc, z_g, E_g, H0)
        return float(chi2)
    return loss

def fit_lcdm(sn: Dict, desi: Dict, cc: Dict, zmax: float,
             use_sn: bool, use_bao: bool, use_cc: bool, rng: np.random.Generator) -> FitResult:
    # LCDM: d_late=0; zt irrelevant
    z_g, I_g, E_g = background("sigmoid", 0.0, 1.0, 0.10, zmax)

    if use_bao:
        bounds = [CFG.H0_BOUNDS, CFG.RD_BOUNDS]
        x0 = np.array([70.0, 147.0], float)
        jitter = np.array([1.0, 2.5], float)

        def loss(p):
            H0, rd = float(p[0]), float(p[1])
            chi2 = 0.0
            if use_sn:  chi2 += chi2_sn(sn, z_g, I_g, H0)
            if use_bao: chi2 += chi2_bao(desi, z_g, I_g, E_g, H0, rd)
            if use_cc:  chi2 += chi2_cc(cc, z_g, E_g, H0)
            return float(chi2)

        best = minimize_multistart(loss, x0, bounds, rng, CFG.N_RESTARTS_GLOBAL, jitter)
        x4 = np.array([0.0, 1.0, best.x[0], best.x[1]], float)
        return FitResult(best.success, best.fun, x4, best.nit, best.status, best.message, best.method, best.n_success_starts)

    bounds = [CFG.H0_BOUNDS]
    x0 = np.array([70.0], float)
    jitter = np.array([1.0], float)

    def loss(p):
        H0 = float(p[0])
        chi2 = 0.0
        if use_sn: chi2 += chi2_sn(sn, z_g, I_g, H0)
        if use_cc: chi2 += chi2_cc(cc, z_g, E_g, H0)
        return float(chi2)

    best = minimize_multistart(loss, x0, bounds, rng, CFG.N_RESTARTS_GLOBAL, jitter)
    x4 = np.array([0.0, 1.0, best.x[0], 147.0], float)
    return FitResult(best.success, best.fun, x4, best.nit, best.status, best.message, best.method, best.n_success_starts)

def fit_late(model: str, width: float, sn: Dict, desi: Dict, cc: Dict, zmax: float,
             use_sn: bool, use_bao: bool, use_cc: bool, rng: np.random.Generator,
             x0: Optional[np.ndarray] = None, n_starts: Optional[int] = None) -> FitResult:
    loss = make_late_loss(model, width, sn, desi, cc, zmax, use_sn, use_bao, use_cc)
    bounds = [CFG.DLATE_BOUNDS, CFG.ZT_BOUNDS, CFG.H0_BOUNDS, CFG.RD_BOUNDS]
    if x0 is None:
        x0 = np.array([0.0, 0.05, 70.0, 147.0], float)
    jitter = np.array([0.15, 0.05, 1.0, 2.5], float)
    if n_starts is None:
        n_starts = CFG.N_RESTARTS_GLOBAL
    return minimize_multistart(loss, x0, bounds, rng, n_starts, jitter)


# =============================================================================
# 7) FIG1/FIG2 constant-delta scans
# =============================================================================

def constant_delta_background(deltaA: float, zmax: float, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.linspace(0.0, float(zmax), int(n))
    rho_de = (1.0 + z)**(3.0 * float(deltaA))
    O_de = 1.0 - float(CFG.Om_m0) - float(CFG.Om_r0)
    Esq = float(CFG.Om_m0) * (1 + z)**3 + float(CFG.Om_r0) * (1 + z)**4 + O_de * rho_de
    E = np.sqrt(np.maximum(Esq, 1e-14))
    dz = z[1] - z[0]
    invE = 1.0 / E
    I = np.zeros_like(z)
    I[1:] = np.cumsum(0.5 * (invE[1:] + invE[:-1])) * dz
    return z, I, E

def fig1_fig2_scans(sn: Dict, desi: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(">>> Fig1/Fig2 scans (SN+BAO), constant delta ansatz")
    dA_vals = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.17, 0.2, 0.3], float)
    res1 = []

    rd0 = np.array([138.0], float)
    for dA in dA_vals:
        z_g, I_g, E_g = constant_delta_background(float(dA), zmax=CFG.FIG12_ZMAX, n=CFG.FIG12_N)

        def loss_rd(x):
            rd = float(x[0])
            if not (130.0 <= rd <= 170.0): return 1e30
            return (chi2_sn(sn, z_g, I_g, CFG.H0_PHYS_SCAN) +
                    chi2_bao(desi, z_g, I_g, E_g, CFG.H0_PHYS_SCAN, rd))

        r = minimize(loss_rd, rd0, bounds=[(130, 170)], method="L-BFGS-B")
        if r.success:
            rd0 = np.array(r.x, float)
        res1.append({"dA": float(dA), "chi2": float(r.fun), "rd": float(r.x[0]), "success": bool(r.success)})

    D_vals = np.unique(np.concatenate([np.linspace(10.0, 10.8, 6), np.linspace(10.8, 12.0, 7)])).astype(float)
    res2 = []
    p0 = np.array([10.0, 138.0], float)  # k, rd
    for D in D_vals:
        micro = 1.0 - float(D)/11.0

        def loss_kr(x):
            k, rd = float(x[0]), float(x[1])
            if not (0.1 <= k <= 20.0): return 1e30
            if not (130.0 <= rd <= 170.0): return 1e30
            deltaA = k * micro
            z_g, I_g, E_g = constant_delta_background(float(deltaA), zmax=CFG.FIG12_ZMAX, n=CFG.FIG12_N)
            return (chi2_sn(sn, z_g, I_g, CFG.H0_PHYS_SCAN) +
                    chi2_bao(desi, z_g, I_g, E_g, CFG.H0_PHYS_SCAN, rd))

        r = minimize(loss_kr, p0, bounds=[(0.1, 20.0), (130, 170)], method="L-BFGS-B")
        if r.success:
            p0 = np.array(r.x, float)
        res2.append({"D": float(D), "chi2": float(r.fun), "k": float(r.x[0]), "rd": float(r.x[1]), "success": bool(r.success)})

    return pd.DataFrame(res1), pd.DataFrame(res2)


# =============================================================================
# 8) 2D PROFILE LIKELIHOOD (warm-start) — PRECOMPUTE BACKGROUND PER CELL [FIX-3]
# =============================================================================

def profile_2d(model: str, width: float, sn: Dict, desi: Dict, cc: Dict, zmax: float,
               seed_base: int) -> pd.DataFrame:
    d_grid = np.linspace(CFG.DLATE_GRID_MIN, CFG.DLATE_GRID_MAX, CFG.N_GRID)
    u_grid = np.linspace(np.log(CFG.ZT_GRID_MIN), np.log(CFG.ZT_GRID_MAX), CFG.N_GRID)
    zt_grid = np.exp(u_grid)

    best_H0 = np.full((CFG.N_GRID, CFG.N_GRID), np.nan)
    best_rd = np.full((CFG.N_GRID, CFG.N_GRID), np.nan)
    best_chi = np.full((CFG.N_GRID, CFG.N_GRID), np.nan)
    best_ok = np.full((CFG.N_GRID, CFG.N_GRID), False, dtype=bool)
    best_nit = np.full((CFG.N_GRID, CFG.N_GRID), -1, dtype=int)
    best_method = np.full((CFG.N_GRID, CFG.N_GRID), "", dtype=object)

    for i, d_late in enumerate(d_grid):
        js = range(CFG.N_GRID) if (i % 2 == 0) else reversed(range(CFG.N_GRID))
        for j in js:
            zt = float(zt_grid[j])
            rng = np.random.default_rng(int(seed_base + i * 10000 + j))

            # PRECOMPUTE background ONCE per cell
            z_g, I_g, E_g = background(model, float(d_late), float(zt), float(width), float(zmax))

            def fun(p):
                H0, rd = float(p[0]), float(p[1])
                if not (CFG.H0_BOUNDS[0] <= H0 <= CFG.H0_BOUNDS[1]): return 1e30
                if not (CFG.RD_BOUNDS[0] <= rd <= CFG.RD_BOUNDS[1]): return 1e30

                # [FIX-B] robust if BAO becomes empty after zcut
                chi = chi2_sn(sn, z_g, I_g, H0) + chi2_cc(cc, z_g, E_g, H0)
                if len(desi["z"]) > 0:
                    chi += chi2_bao(desi, z_g, I_g, E_g, H0, rd)
                return float(chi)

            warm = []
            if i > 0 and np.isfinite(best_H0[i-1, j]) and np.isfinite(best_rd[i-1, j]):
                warm.append(np.array([best_H0[i-1, j], best_rd[i-1, j]], float))
            if j > 0 and np.isfinite(best_H0[i, j-1]) and np.isfinite(best_rd[i, j-1]):
                warm.append(np.array([best_H0[i, j-1], best_rd[i, j-1]], float))

            x0 = warm[0] if warm else np.array([72.0, 147.0], float)
            n_starts = max(CFG.MIN_RESTARTS_WARM, CFG.N_RESTARTS_H0RD // 2) if warm else CFG.N_RESTARTS_H0RD
            jitter = np.array([1.0, 2.0], float)

            best = minimize_multistart(fun, x0, [CFG.H0_BOUNDS, CFG.RD_BOUNDS], rng, n_starts, jitter)
            best_H0[i, j] = float(best.x[0])
            best_rd[i, j] = float(best.x[1])
            best_chi[i, j] = float(best.fun)
            best_ok[i, j] = bool(best.success)
            best_nit[i, j] = int(best.nit)
            best_method[i, j] = str(best.method)

    rows = []
    for i, d_late in enumerate(d_grid):
        for j, zt in enumerate(zt_grid):
            rows.append({
                "model": model,
                "width": float(width),
                "d_late": float(d_late),
                "z_trans": float(zt),
                "chi2": float(best_chi[i, j]),
                "H0": float(best_H0[i, j]),
                "rd": float(best_rd[i, j]),
                "success": bool(best_ok[i, j]),
                "nit": int(best_nit[i, j]),
                "method": str(best_method[i, j]),
            })
    return pd.DataFrame(rows)

def bestfit_from_profile(df2d: pd.DataFrame) -> np.ndarray:
    row = df2d.loc[df2d["chi2"].idxmin()]
    return np.array([row["d_late"], row["z_trans"], row["H0"], row["rd"]], float)


# =============================================================================
# 9) ABLATIONS + COMPARISON TABLES
# =============================================================================

def run_ablations(model: str, width: float, sn: Dict, desi: Dict, cc: Dict, zmax: float,
                  rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = [
        ("ALL",    True, True, True),
        ("SN+BAO", True, True, False),
        ("SN+CC",  True, False, True),
        ("BAO+CC", False, True, True),
    ]

    late_rows, lcdm_rows, comp_rows = [], [], []

    for name, usn, ubao, ucc in cases:
        rS = fit_late(model, width, sn, desi, cc, zmax, usn, ubao, ucc, rng)
        r0 = fit_lcdm(sn, desi, cc, zmax, usn, ubao, ucc, rng)

        late_rows.append({
            "case": name, "model": model, "width": float(width),
            "success": rS.success, "chi2": rS.fun,
            "d_late": float(rS.x[0]), "z_trans": float(rS.x[1]),
            "H0": float(rS.x[2]), "rd": float(rS.x[3]),
            "nit": rS.nit, "status": rS.status, "method": rS.method,
            "n_success_starts": rS.n_success_starts,
            "message": rS.message
        })
        lcdm_rows.append({
            "case": name, "model": "lcdm", "width": float(width),
            "success": r0.success, "chi2": r0.fun,
            "H0": float(r0.x[2]), "rd": float(r0.x[3]),
            "nit": r0.nit, "status": r0.status, "method": r0.method,
            "n_success_starts": r0.n_success_starts,
            "message": r0.message
        })

        n = n_points(usn, ubao, ucc, sn, desi, cc)
        dchi = float(r0.fun - rS.fun)

        kS = k_params_late(ubao)
        k0 = k_params_lcdm(ubao)

        # [FIX-6] non-regular case: provide both conservative dof=1 and heuristic dof=2
        comp_rows.append({
            "case": name, "model": model, "width": float(width),
            "chi2_late": float(rS.fun), "chi2_lcdm": float(r0.fun),
            "delta_chi2": dchi,
            "p_lrt_dof1_conservative": lrt_pvalue_heuristic(dchi, dof=1),
            "p_lrt_dof2_heuristic": lrt_pvalue_heuristic(dchi, dof=2),
            "n": int(n),
            "k_late": int(kS),
            "k_lcdm": int(k0),
            "dAIC": aic(rS.fun, kS) - aic(r0.fun, k0),
            "dBIC": bic(rS.fun, kS, n) - bic(r0.fun, k0, n),
        })

        logger.info(
            f"  [{model} w={width:.2f}] {name:6s} "
            f"χ²={rS.fun:.3f} (lcdm={r0.fun:.3f}) Δχ²={dchi:.3f} "
            f"p(dof1)={comp_rows[-1]['p_lrt_dof1_conservative']:.3g} "
            f"p(dof2)={comp_rows[-1]['p_lrt_dof2_heuristic']:.3g} "
            f"H0={rS.x[2]:.2f} rd={rS.x[3]:.2f} zt={rS.x[1]:.3f} d={rS.x[0]:+.3f}"
        )

    return pd.DataFrame(late_rows), pd.DataFrame(lcdm_rows), pd.DataFrame(comp_rows)


# =============================================================================
# 10) PPC DIAGNOSTICS (residuals + pulls) — CHOLESKY WHITENING [FIX-4]
# =============================================================================

def _whiten_pulls_from_cov(res: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Compute whitened pulls y = L^{-1} res where C = L L^T.
    If model is correct and C is correct, y ~ N(0, I).
    """
    res = np.asarray(res, float)
    if res.size == 0:
        return res
    L = np.linalg.cholesky(C)
    y = np.linalg.solve(L, res)
    return y

def ppc(tag: str, model: str, width: float, zmax: float,
        theta4: np.ndarray, sn: Dict, desi: Dict, cc: Dict,
        use_sn: bool, use_bao: bool, use_cc: bool) -> None:
    d_late, zt, H0, rd = [float(v) for v in theta4]
    z_g, I_g, E_g = background(model, d_late, zt, width, zmax)

    # SN
    if use_sn:
        z = sn["z"]
        I = safe_interp(z, z_g, I_g, "PPC SN I(z)")
        dl = np.maximum((1.0 + z) * (CFG.c_light / H0) * I, 1e-30)
        mu = 5.0 * np.log10(dl) + 25.0
        dvec = mu - sn["mB"]
        nuis = -(sn["P"] @ dvec)
        g = sn["glob"]
        res = dvec + (nuis[0]*g[0] + nuis[1]*g[1] + nuis[2]*g[2])

        pull_white = _whiten_pulls_from_cov(res, sn["C"])
        pd.DataFrame({"z": z, "residual": res, "pull_white": pull_white}).to_csv(
            os.path.join(CFG.OUTDIR, f"ppc_{tag}_sn.csv"), index=False
        )

    # BAO
    if use_bao and len(desi["z"]) > 0:
        z = desi["z"]
        I = safe_interp(z, z_g, I_g, "PPC BAO I(z)")
        E = safe_interp(z, z_g, E_g, "PPC BAO E(z)")
        DM = (CFG.c_light / H0) * I
        DH = CFG.c_light / (H0 * E)
        DV = (z * DH * DM**2)**(1.0/3.0)
        tm = desi["tm"]
        model_obs = np.zeros_like(z)
        model_obs[tm == 0] = DV[tm == 0]
        model_obs[tm == 1] = DM[tm == 1]
        model_obs[tm == 2] = DH[tm == 2]
        pred = model_obs / rd
        res = pred - desi["y"]

        pull_white = _whiten_pulls_from_cov(res, desi["C"])
        pd.DataFrame({
            "z": z, "type_mask": tm, "y": desi["y"], "pred": pred,
            "residual": res, "pull_white": pull_white
        }).to_csv(os.path.join(CFG.OUTDIR, f"ppc_{tag}_bao.csv"), index=False)

    # CC
    if use_cc and len(cc["z"]) > 0:
        z = cc["z"]
        Ez = safe_interp(z, z_g, E_g, "PPC CC E(z)")
        pred = H0 * Ez
        res = pred - cc["Hz"]
        pull = res / cc["sig"]
        pd.DataFrame({"z": z, "Hz": cc["Hz"], "pred": pred, "residual": res, "pull": pull}).to_csv(
            os.path.join(CFG.OUTDIR, f"ppc_{tag}_cc.csv"), index=False
        )


# =============================================================================
# 11) PLOTTING
# =============================================================================

def plot_fig1(df1: pd.DataFrame, tag: str) -> None:
    d1 = df1.sort_values("dA")
    dA = d1["dA"].values.astype(float)
    chi = d1["chi2"].values.astype(float)
    rd = d1["rd"].values.astype(float)
    dchi = chi - chi.min()

    x = np.linspace(dA.min(), dA.max(), 400)
    spl_c = PchipInterpolator(dA, dchi)
    spl_r = PchipInterpolator(dA, rd)

    fig, ax = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(x, spl_c(x), lw=2)
    ax[0].plot(dA, dchi, "o", alpha=0.75, ms=4)
    ax[0].axvspan(dA.min(), 0.0, alpha=0.1)
    ax[0].axvline(0.0, ls=":", lw=1)
    ax[0].axhline(1.0, ls="--", lw=0.8, alpha=0.35)
    ax[0].axhline(4.0, ls="--", lw=0.8, alpha=0.35)
    ax[0].set_ylabel(r"$\Delta\chi^2$")

    ax[1].plot(x, spl_r(x), lw=2)
    ax[1].axhline(147.1, ls=":", lw=1)
    ax[1].set_ylabel(r"$r_d$ [Mpc]")
    ax[1].set_xlabel(r"Geometric Anomaly $\delta_A$")
    save_figure(fig, f"Figure_1_Diagnosis_{tag}")

def plot_fig2(df2: pd.DataFrame, tag: str) -> None:
    d2 = df2.sort_values("D")
    D = d2["D"].values.astype(float)
    chi = d2["chi2"].values.astype(float)
    dchi = chi - chi.min()
    x = np.linspace(D.min(), D.max(), 400)
    spl = PchipInterpolator(D, dchi)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(x, spl(x), "--", lw=1.5)
    ax.plot(D, dchi, "ko", ms=3)
    ax.set_ylabel(r"$\Delta\chi^2$")
    ax.set_xlabel(r"Effective Dimension $D$")
    save_figure(fig, f"Figure_2_Control_{tag}")

def plot_profile2d(df2d: pd.DataFrame, best: np.ndarray, tag: str) -> None:
    d_v = np.sort(df2d["d_late"].unique())
    z_v = np.sort(df2d["z_trans"].unique())
    Z, Dm = np.meshgrid(z_v, d_v)
    C = np.full_like(Z, np.nan, dtype=float)

    for i, dv in enumerate(d_v):
        sub = df2d[df2d["d_late"] == dv]
        mapping = {float(r["z_trans"]): float(r["chi2"]) for _, r in sub.iterrows()}
        for j, zv in enumerate(z_v):
            if float(zv) in mapping:
                C[i, j] = mapping[float(zv)]
    C = C - np.nanmin(C)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    # [FIX-A] colormap portability
    ax.contourf(Z, Dm, C, levels=20, cmap=plt.get_cmap("viridis").reversed())
    cs = ax.contour(Z, Dm, C, levels=[2.3, 6.18], colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fontsize=8, fmt={2.3: "1σ", 6.18: "2σ"})
    ax.plot(float(best[1]), float(best[0]), "r*", ms=10, mec="white")
    ax.set_xlabel(r"$z_{\rm trans}$")
    ax.set_ylabel(r"$\delta_{\rm late}$")
    ax.set_xscale("log")
    save_figure(fig, f"Figure_4_Profile_{tag}")

def plot_money(model: str, width: float, zmax: float,
               best: np.ndarray, sn: Dict, desi: Dict, cc: Dict, tag: str) -> None:
    d_late, zt, H0_alt, rd_alt = [float(v) for v in best]

    rng = np.random.default_rng(CFG.SEED + 999)
    lcdm_cc = fit_lcdm(sn, desi, cc, zmax, use_sn=False, use_bao=False, use_cc=True, rng=rng)
    H0_null = float(lcdm_cc.x[2])

    zmax_plot = min(1.5, float(np.max(cc["z"]) if len(cc["z"]) else 1.5))
    z = np.linspace(0.0, zmax_plot, 400)

    z_g, I_g, E_g = background(model, d_late, zt, width, zmax)
    H_alt = H0_alt * safe_interp(z, z_g, E_g, "Money E(z)")

    z0, I0, E0 = background("sigmoid", 0.0, 1.0, 0.10, zmax)
    H_null = H0_null * safe_interp(z, z0, E0, "Money LCDM E(z)")

    if model == "sigmoid":
        w_eff = -1.0 + delta_sigmoid(z, d_late, zt, width)
    elif model == "tanh":
        w_eff = -1.0 + delta_tanh(z, d_late, zt, width)
    else:
        w_eff = -1.0 + delta_piecewise(z, d_late, zt, width)

    fig, ax = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    if len(cc["z"]) > 0:
        ax[0].errorbar(cc["z"], cc["Hz"], yerr=cc["sig"], fmt="o", color="k", alpha=0.3, ms=2, elinewidth=1)
    ax[0].plot(z, H_null, "--", color="gray", lw=1.5, label=r"$\Lambda$CDM (CC-fit)")
    ax[0].plot(z, H_alt, "-", lw=2.0, label=f"{model} best-fit")
    ax[0].axvline(zt, color="orange", ls=":", lw=1)
    ax[0].set_ylabel(r"$H(z)$ [km/s/Mpc]")
    ax[0].legend(frameon=False)

    ax[0].text(
        0.04, 0.06,
        rf"$H_0={H0_alt:.2f}$"+"\n"+rf"$r_d={rd_alt:.2f}$"+"\n"+rf"$z_t={zt:.3f}$"+"\n"+rf"$\delta={d_late:+.3f}$",
        transform=ax[0].transAxes, fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0.5)
    )

    ax[1].plot(z, w_eff, lw=2)
    ax[1].axhline(-1.0, color="gray", ls="--", lw=1)
    ax[1].fill_between(z, -1.0, w_eff, where=(w_eff < -1.0), alpha=0.1)
    ax[1].set_ylabel(r"$w_{\rm eff}(z)$")
    ax[1].set_xlabel(r"Redshift $z$")
    ax[1].axvline(zt, color="orange", ls=":", lw=1)
    save_figure(fig, f"Figure_3_Money_{tag}")


# =============================================================================
# 12) MCMC (Metropolis-Hastings) — unchanged logic, safer priors
# =============================================================================

def log_prior(theta: np.ndarray, use_bao: bool) -> float:
    d_late, zt, H0, rd = [float(v) for v in theta]
    if not (CFG.DLATE_BOUNDS[0] <= d_late <= CFG.DLATE_BOUNDS[1]): return -np.inf
    if not (CFG.ZT_BOUNDS[0] <= zt <= CFG.ZT_BOUNDS[1]): return -np.inf
    if not (CFG.H0_BOUNDS[0] <= H0 <= CFG.H0_BOUNDS[1]): return -np.inf
    if use_bao and (not (CFG.RD_BOUNDS[0] <= rd <= CFG.RD_BOUNDS[1])): return -np.inf
    return 0.0

def mh_mcmc(logpost: Callable[[np.ndarray], float],
            x0: np.ndarray,
            steps: int,
            burn: int,
            thin: int,
            proposal_scales: np.ndarray,
            seed: int) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(int(seed))
    x = np.array(x0, float)
    lp = float(logpost(x))

    dim = x.size
    chain = []
    accepts = 0
    scale = np.array(proposal_scales, float)

    for t in range(int(steps)):
        x_prop = x + rng.normal(0.0, scale, size=dim)
        lp_prop = float(logpost(x_prop))

        if np.isfinite(lp_prop):
            if np.log(rng.random()) < (lp_prop - lp):
                x = x_prop
                lp = lp_prop
                accepts += 1

        if t < burn and (t + 1) % 500 == 0:
            acc_rate = accepts / max(t + 1, 1)
            if acc_rate < 0.18:
                scale *= 0.85
            elif acc_rate > 0.35:
                scale *= 1.15

        if t >= burn and ((t - burn) % thin == 0):
            chain.append(np.concatenate([x, [lp]]))

    chain = np.array(chain, float)
    info = {
        "steps": int(steps),
        "burn": int(burn),
        "thin": int(thin),
        "accept_rate": float(accepts / max(steps, 1)),
        "final_scales": scale.tolist(),
    }
    return chain, info

def corner_plot(chain: np.ndarray, labels: List[str], tag: str) -> None:
    X = chain[:, :-1]
    d = X.shape[1]
    fig = plt.figure(figsize=(2.2 * d, 2.2 * d))
    gs = fig.add_gridspec(d, d, wspace=0.05, hspace=0.05)

    for i in range(d):
        for j in range(d):
            ax = fig.add_subplot(gs[i, j])
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(X[:, j], bins=40, histtype="step")
                ax.set_yticks([])
            else:
                ax.hist2d(X[:, j], X[:, i], bins=40)
            if i == d - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticks([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i])
            else:
                if j != 0:
                    ax.set_yticks([])
    save_figure(fig, f"corner_{tag}")

def summarize_chain(chain: np.ndarray, labels: List[str]) -> Dict:
    X = chain[:, :-1]
    out = {}
    for i, lab in enumerate(labels):
        q16, q50, q84 = np.percentile(X[:, i], [16, 50, 84])
        out[lab] = {"p16": float(q16), "p50": float(q50), "p84": float(q84), "sigma": float(0.5*(q84-q16))}
    out["logpost_max"] = float(np.max(chain[:, -1]))
    return out


# =============================================================================
# 13) MAIN PIPELINE PER zcut
# =============================================================================

def run_one_zcut(zcut: float, models: List[str], widths: List[float],
                 do_mcmc: bool, mcmc_cases: List[str], mcmc_steps: int, mcmc_burn: int, mcmc_thin: int) -> None:
    sn, desi, cc = load_data(zcut)
    zmax = infer_zmax(sn, desi, cc)

    logger.info(f"\n==================== RUN zcut={zcut:.1f} ====================")
    logger.info(f"SN N={len(sn['z'])} | BAO N={len(desi['z'])} | CC N={len(cc['z'])} | inferred zmax={zmax:.3f}")

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": asdict(CFG),
        "run_args": {
            "zcut": float(zcut),
            "models": models,
            "widths": widths,
            "do_mcmc": bool(do_mcmc),
            "mcmc_cases": mcmc_cases,
            "mcmc_steps": int(mcmc_steps),
            "mcmc_burn": int(mcmc_burn),
            "mcmc_thin": int(mcmc_thin),
        },
        "data_summary": {
            "N_sn": int(len(sn["z"])),
            "N_bao": int(len(desi["z"])),
            "N_cc": int(len(cc["z"])),
            "zmax_sn": float(np.max(sn["z"])),
            "zmax_bao": float(np.max(desi["z"])) if len(desi["z"]) else 0.0,
            "zmax_cc": float(np.max(cc["z"])) if len(cc["z"]) else 0.0,
            "zmax_inferred": float(zmax),
        }
    }
    save_json(os.path.join(CFG.OUTDIR, f"run_metadata_zcut{zcut:.1f}.json"), meta)

    # Fig1/Fig2
    df1, df2 = fig1_fig2_scans(sn, desi)
    df1.to_csv(os.path.join(CFG.OUTDIR, f"fig1_scan_zcut{zcut:.1f}.csv"), index=False)
    df2.to_csv(os.path.join(CFG.OUTDIR, f"fig2_scan_zcut{zcut:.1f}.csv"), index=False)
    plot_fig1(df1, tag=f"zcut{zcut:.1f}")
    plot_fig2(df2, tag=f"zcut{zcut:.1f}")

    width_records = []
    rng = np.random.default_rng(int(CFG.SEED + int(round(zcut * 1000))))

    for width in widths:
        for model in models:
            logger.info(f"\n--- zcut={zcut:.1f} model={model} width={width:.2f} ---")

            seed_base = int(CFG.SEED + int(round(zcut * 1000)) + int(round(width * 10000)) + MODEL_SEED_TAG[model])  # [FIX-1]
            df2d = profile_2d(model, float(width), sn, desi, cc, zmax, seed_base=seed_base)
            prof_path = os.path.join(CFG.OUTDIR, f"profile2d_{model}_w{width:.2f}_zcut{zcut:.1f}.csv")
            df2d.to_csv(prof_path, index=False)

            best = bestfit_from_profile(df2d)
            width_records.append({"zcut": float(zcut), "model": model, "width": float(width), "chi2_min": float(df2d["chi2"].min())})

            # Ablations & comparison
            dfS, df0, dfC = run_ablations(model, float(width), sn, desi, cc, zmax, rng)
            dfS["zcut"] = float(zcut); df0["zcut"] = float(zcut); dfC["zcut"] = float(zcut)

            dfS.to_csv(os.path.join(CFG.OUTDIR, f"ablations_{model}_w{width:.2f}_zcut{zcut:.1f}.csv"), index=False)
            df0.to_csv(os.path.join(CFG.OUTDIR, f"lcdm_{model}_w{width:.2f}_zcut{zcut:.1f}.csv"), index=False)
            dfC.to_csv(os.path.join(CFG.OUTDIR, f"comparison_{model}_w{width:.2f}_zcut{zcut:.1f}.csv"), index=False)

            # Figures (Money + Profile2D)
            tag = f"{model}_w{width:.2f}_zcut{zcut:.1f}"
            plot_money(model, float(width), zmax, best, sn, desi, cc, tag=tag)
            plot_profile2d(df2d, best, tag=tag)

            # PPC diagnostics for ALL using 2D best-fit
            ppc(tag=f"{tag}_ALL_best2d", model=model, width=float(width), zmax=zmax, theta4=best,
                sn=sn, desi=desi, cc=cc, use_sn=True, use_bao=True, use_cc=True)

            # Optional MCMC
            if do_mcmc:
                for case in mcmc_cases:
                    case = case.strip().upper()
                    if case not in ["ALL", "SN+BAO", "SN+CC", "BAO+CC"]:
                        continue
                    use_sn = case in ["ALL", "SN+BAO", "SN+CC"]
                    use_bao = case in ["ALL", "SN+BAO", "BAO+CC"]
                    use_cc_ = case in ["ALL", "SN+CC", "BAO+CC"]

                    r_init = fit_late(model, float(width), sn, desi, cc, zmax, use_sn, use_bao, use_cc_, rng,
                                       x0=best, n_starts=max(6, CFG.N_RESTARTS_GLOBAL // 2))
                    x0 = np.array(r_init.x, float)

                    loss = make_late_loss(model, float(width), sn, desi, cc, zmax, use_sn, use_bao, use_cc_)
                    def logpost(x):
                        lp = log_prior(x, use_bao=use_bao)
                        if not np.isfinite(lp):
                            return -np.inf
                        return lp - 0.5 * float(loss(x))

                    scales = np.array([0.06, 0.02, 0.6, 1.8], float)
                    if not use_bao:
                        scales[3] = 1e-6

                    mtag = f"{model}_w{width:.2f}_zcut{zcut:.1f}_{case}"
                    seed = int(CFG.SEED + 777 + int(round(zcut * 1000)) + int(round(width * 10000)) + MODEL_SEED_TAG[model])
                    chain, info = mh_mcmc(
                        logpost, x0,
                        steps=mcmc_steps, burn=mcmc_burn, thin=mcmc_thin,
                        proposal_scales=scales,
                        seed=seed
                    )

                    np.save(os.path.join(CFG.OUTDIR, f"mcmc_chain_{mtag}.npy"), chain)
                    labels = [r"$\delta_{\rm late}$", r"$z_{\rm trans}$", r"$H_0$", r"$r_d$"]
                    summ = summarize_chain(chain, labels)
                    summ.update({"mcmc_info": info, "init": x0.tolist(), "chi2_init": float(loss(x0))})
                    save_json(os.path.join(CFG.OUTDIR, f"mcmc_summary_{mtag}.json"), summ)
                    corner_plot(chain, labels=labels, tag=mtag)

                    logger.info(f"    [MCMC] {mtag}: accept={info['accept_rate']:.3f} saved chain with {len(chain)} samples")

    # Width sensitivity plot
    dfw = pd.DataFrame(width_records)
    dfw.to_csv(os.path.join(CFG.OUTDIR, f"width_sensitivity_zcut{zcut:.1f}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for model in sorted(dfw["model"].unique()):
        sub = dfw[dfw["model"] == model].sort_values("width")
        w = sub["width"].values.astype(float)
        dchi = sub["chi2_min"].values.astype(float) - float(np.min(sub["chi2_min"].values.astype(float)))
        ax.plot(w, dchi, marker="o", label=model)
    ax.set_xlabel("width")
    ax.set_ylabel(r"$\Delta\chi^2_{\min}(w)$")
    ax.legend(frameon=False)
    ax.set_title(f"Width sensitivity (zcut={zcut:.1f})")
    save_figure(fig, f"Figure_width_profile_zcut{zcut:.1f}")

    logger.info(f"\n[OK] Completed zcut={zcut:.1f}\n")


# =============================================================================
# 14) PLOT-ONLY MODE
# =============================================================================

def plot_only() -> None:
    logger.info(">>> Plot-only mode: regenerating figures from existing CSVs.")
    files = os.listdir(CFG.OUTDIR)

    for fn in files:
        if fn.startswith("fig1_scan_zcut") and fn.endswith(".csv"):
            tag = fn.replace(".csv", "").replace("fig1_scan_", "")
            ztag = tag.replace("zcut", "")
            p1 = os.path.join(CFG.OUTDIR, fn)
            p2 = os.path.join(CFG.OUTDIR, f"fig2_scan_zcut{ztag}.csv")
            if os.path.exists(p2):
                df1 = pd.read_csv(p1)
                df2 = pd.read_csv(p2)
                plot_fig1(df1, tag=tag)
                plot_fig2(df2, tag=tag)

    for fn in files:
        if fn.startswith("width_sensitivity_zcut") and fn.endswith(".csv"):
            zcut = float(fn.replace("width_sensitivity_zcut", "").replace(".csv", ""))
            dfw = pd.read_csv(os.path.join(CFG.OUTDIR, fn))
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            for model in sorted(dfw["model"].unique()):
                sub = dfw[dfw["model"] == model].sort_values("width")
                w = sub["width"].values.astype(float)
                dchi = sub["chi2_min"].values.astype(float) - float(np.min(sub["chi2_min"].values.astype(float)))
                ax.plot(w, dchi, marker="o", label=model)
            ax.set_xlabel("width")
            ax.set_ylabel(r"$\Delta\chi^2_{\min}(w)$")
            ax.legend(frameon=False)
            ax.set_title(f"Width sensitivity (zcut={zcut:.1f})")
            save_figure(fig, f"Figure_width_profile_zcut{zcut:.1f}_PLOTONLY")

    profs = [f for f in files if f.startswith("profile2d_") and f.endswith(".csv")]
    for pf in profs:
        base = pf.replace(".csv", "")
        parts = base.split("_")
        if len(parts) < 4:
            continue
        model = parts[1]
        w_str = parts[2].replace("w", "")
        z_str = parts[3].replace("zcut", "")
        try:
            width = float(w_str)
            zcut = float(z_str)
        except Exception:
            continue

        sn, desi, cc = load_data(zcut)
        zmax = infer_zmax(sn, desi, cc)

        df2d = pd.read_csv(os.path.join(CFG.OUTDIR, pf))
        best = bestfit_from_profile(df2d)
        tag = f"{model}_w{width:.2f}_zcut{zcut:.1f}_PLOTONLY"
        plot_money(model, width, zmax, best, sn, desi, cc, tag=tag)
        plot_profile2d(df2d, best, tag=tag)

    logger.info("[OK] Plot-only completed.")


# =============================================================================
# 15) CLI
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="late Journal-Grade Ultra-Robust Pipeline (PRD/JCAP) — FIXED")
    ap.add_argument("--mode", choices=["run", "plot"], default="run")
    ap.add_argument("--zcuts", nargs="*", type=float, default=[1.5, 1.2, 1.0])
    ap.add_argument("--widths", nargs="*", type=float, default=list(CFG.WIDTH_LIST))
    ap.add_argument("--models", nargs="*", type=str, default=list(CFG.MODEL_LIST))

    ap.add_argument("--do_mcmc", action="store_true", help="Run built-in MH-MCMC for selected cases")
    ap.add_argument("--mcmc_cases", nargs="*", type=str, default=["ALL", "SN+CC"], help="Cases: ALL SN+BAO SN+CC BAO+CC")
    ap.add_argument("--mcmc_steps", type=int, default=CFG.MCMC_STEPS)
    ap.add_argument("--mcmc_burn", type=int, default=CFG.MCMC_BURN)
    ap.add_argument("--mcmc_thin", type=int, default=CFG.MCMC_THIN)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(CFG.OUTDIR, exist_ok=True)

    if args.mode == "plot":
        plot_only()
        return

    models = [m.strip().lower() for m in args.models]
    for m in models:
        if m not in ("sigmoid", "tanh", "piecewise"):
            raise ValueError(f"Unknown model: {m}")

    widths = [float(w) for w in args.widths]
    zcuts = [float(z) for z in args.zcuts]

    for zcut in zcuts:
        run_one_zcut(
            zcut=zcut,
            models=models,
            widths=widths,
            do_mcmc=bool(args.do_mcmc),
            mcmc_cases=list(args.mcmc_cases),
            mcmc_steps=int(args.mcmc_steps),
            mcmc_burn=int(args.mcmc_burn),
            mcmc_thin=int(args.mcmc_thin),
        )

    logger.info("\nALL DONE.\n")

if __name__ == "__main__":
    main()
