# %%
# Peak (start-of-fill) luminosity scan: no time evolution and no leveling, just the
# luminosity integrator of lumi_class.LumiModel evaluated at a fixed beam state.
# Crossing angle and crab cavities are both in the horizontal plane. Intensity,
# emittance and bunch length are kept constant. Figures:
#   - HL-LHC and LHC reference case: lumi vs sigma*_x, one curve per sigma*_y
#   - HL-LHC: lumi vs beta*_x, one curve per beta*_x/beta*_y, with the effective Piwinski angle on top
#   - HL-LHC: lumi and angles vs beta*_x at fixed beta*_y, both beams / B1 only / no crab
#   - HL-LHC: lumi map vs beta*_x and half crossing angle at fixed beta*_y (angle free, LR separation contours)
#   - HL-LHC: lumi map vs beta*_x/beta*_y and half crossing angle (constant LR separation only)
#
# By default the crossing angle is adjusted so that the normalized long-range
# separation stays constant. Estimate valid far from the IP (s >> beta*), where the
# separation grows as theta_full*s and the beam size as sigma*·s/beta*:
#     d_LR/sigma = theta_full * beta*_x / sigma*_x = theta_full * sigma*_x / eps_geom
#  => theta_full/2 = (d_LR/2) * eps_geom / sigma*_x = (d_LR/2) * sqrt(eps_geom / beta*_x)
# Piwinski angle: phi = (theta_full/2) * sigma_z / sigma*_x (geometric), and with crab
# cavities, from the Gaussian overlap with a linear crab tilt per beam (residual tilts
# a = theta_full/2 - |theta_crab,B1/2|, b = theta_full/2 - |theta_crab,B2/2|, R = 1/sqrt(1 + phi_eff^2)):
#     phi_eff = sqrt((a^2 + b^2) / 2) * sigma_z / sigma*_x
# which is (theta_full/2 - |theta_crab/2|) * sigma_z / sigma*_x for symmetric crabbing and at best
# phi/sqrt(2) with one beam crabbed. phi_eff ignores the hourglass effect and the sinusoidal shape of
# the 400 MHz kick; the luminosity itself always comes from the integrator, which includes both.
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from scipy.constants import c as clight

import lumi_class

warnings.filterwarnings("ignore")

# categorical slots 1-8, plus a marker per curve so curves are not told apart by color alone
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
CASE_LABELS = {"hllhc": "HL-LHC", "lhc_ref": "LHC reference", "hllhc_nocrab": "HL-LHC, no crab",
               "hllhc_crab_b1": "HL-LHC, B1 crab only"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan the peak luminosity vs IP beam size / beta* with the crossing angle in x."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"),
                        help="YAML configuration (energy, emittance, bunch length, bunches, cross sections)")
    parser.add_argument("--intensity", type=float, default=None,
                        help="Bunch intensity [p/bunch] (default: from the configuration)")
    xing = parser.add_mutually_exclusive_group()
    xing.add_argument("--lr-sep-sigma", type=float, default=10.0,
                      help="Long-range separation [sigma] kept constant by adjusting the crossing angle")
    xing.add_argument("--xing-urad", type=float, default=None,
                      help="Fixed half crossing angle in x [urad] (overrides --lr-sep-sigma)")
    parser.add_argument("--crab-urad", type=float, default=-190.0,
                        help="Half crab angle in x [urad] (0 = no crab cavities)")
    parser.add_argument("--no-crab-clip", action="store_true",
                        help="Keep the crab angle even when it exceeds the half crossing angle (over-crabbing)")
    # sigma* scan
    parser.add_argument("--sigmax-min", type=float, default=1.0, help="Minimum sigma*_x [um]")
    parser.add_argument("--sigmax-max", type=float, default=30.0, help="Maximum sigma*_x [um]")
    parser.add_argument("--sigmax-step", type=float, default=1.0, help="sigma*_x step [um]")
    parser.add_argument("--sigmay", type=float, nargs="+", default=[8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                        help="sigma*_y values [um], one curve each")
    # beta* scan (HL-LHC case)
    parser.add_argument("--betax-min", type=float, default=5.0, help="Minimum beta*_x [cm]")
    parser.add_argument("--betax-max", type=float, default=60.0, help="Maximum beta*_x [cm]")
    parser.add_argument("--betax-step", type=float, default=2.5, help="beta*_x step [cm]")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 4.0],
                        help="beta*_x/beta*_y ratios, one curve each")
    # beta*_x scan at fixed beta*_y, HL-LHC with both beams / B1 only / no crab
    parser.add_argument("--fixed-betay", type=float, default=11.0,
                        help="Fixed-beta*_y scan: beta*_y [cm] (negative to disable)")
    parser.add_argument("--fixed-betax-min", type=float, default=5.0, help="Fixed-beta*_y scan: minimum beta*_x [cm]")
    parser.add_argument("--fixed-betax-max", type=float, default=60.0, help="Fixed-beta*_y scan: maximum beta*_x [cm]")
    parser.add_argument("--fixed-betax-step", type=float, default=1.0, help="Fixed-beta*_y scan: beta*_x step [cm]")
    # map at fixed beta*_y: beta*_x vs half crossing angle scanned independently (LR separation not fixed)
    parser.add_argument("--xmap-betay", type=float, default=11.0,
                        help="beta*_x-crossing map: fixed beta*_y [cm] (negative to disable)")
    parser.add_argument("--xmap-betax-min", type=float, default=5.0, help="beta*_x-crossing map: minimum beta*_x [cm]")
    parser.add_argument("--xmap-betax-max", type=float, default=60.0, help="beta*_x-crossing map: maximum beta*_x [cm]")
    parser.add_argument("--xmap-betax-step", type=float, default=1.0, help="beta*_x-crossing map: beta*_x step [cm]")
    parser.add_argument("--xmap-xing-min", type=float, default=50.0,
                        help="beta*_x-crossing map: minimum half crossing angle [urad]")
    parser.add_argument("--xmap-xing-max", type=float, default=350.0,
                        help="beta*_x-crossing map: maximum half crossing angle [urad]")
    parser.add_argument("--xmap-xing-step", type=float, default=5.0,
                        help="beta*_x-crossing map: half crossing angle step [urad]")
    parser.add_argument("--xmap-dsep", type=float, nargs="+", default=[6, 8, 10, 12, 14],
                        help="beta*_x-crossing map: LR separation contours [sigma]")
    parser.add_argument("--xmap-piwinski", type=float, nargs="*", default=[0.25, 0.5, 1, 2, 4],
                        help="beta*_x-crossing map: effective Piwinski angle contours (none to disable)")
    # luminosity map (HL-LHC case)
    parser.add_argument("--map-ratio-min", type=float, default=0.5, help="Map: minimum beta*_x/beta*_y")
    parser.add_argument("--map-ratio-max", type=float, default=5.0, help="Map: maximum beta*_x/beta*_y")
    parser.add_argument("--map-ratio-step", type=float, default=0.1, help="Map: beta*_x/beta*_y step")
    parser.add_argument("--map-xing-min", type=float, default=100.0, help="Map: minimum half crossing angle [urad]")
    parser.add_argument("--map-xing-max", type=float, default=350.0, help="Map: maximum half crossing angle [urad]")
    parser.add_argument("--map-xing-step", type=float, default=5.0, help="Map: half crossing angle step [urad]")
    parser.add_argument("--mark-betay", type=float, default=11.0,
                        help="Map: draw the line of constant beta*_y [cm] (negative to disable)")
    parser.add_argument("--mark-points", nargs="*", default=["18,11", "15,15"],
                        help="Map: stars at these beta*_x,beta*_y [cm] points, e.g. 18,11 15,15 (none to disable)")
    parser.add_argument("--map-cmap", default="magma",
                        help="Map: matplotlib colormap (perceptually uniform: magma, inferno, viridis, cividis...)")
    parser.add_argument("--emittance-um", type=float, default=None,
                        help="Normalized emittance [um] (default: from config)")
    parser.add_argument("--bunch-length-s", type=float, default=None,
                        help="4-sigma bunch length [s] (default: from config)")
    parser.add_argument("--bunches", type=int, default=None,
                        help="Colliding bunches in IP1/5 (default: from config)")
    parser.add_argument("--ref-intensity", type=float, default=1.15e11,
                        help="LHC reference case: bunch intensity [p/bunch]")
    parser.add_argument("--ref-emittance-um", type=float, default=3.5,
                        help="LHC reference case: normalized emittance [um]")
    parser.add_argument("--ref-crab-urad", type=float, default=0.0,
                        help="LHC reference case: half crab angle in x [urad]")
    parser.add_argument("--no-ref", action="store_true", help="Skip the LHC reference case")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="Parallel processes")
    parser.add_argument("--output-dir", type=Path,
                        help="Output folder (default: outputs/peak_lumi_scan_<config filename without extension>)")
    return parser.parse_args()


args = parse_args()
if args.fixed_betay is not None and args.fixed_betay <= 0:
    args.fixed_betay = None
if args.xmap_betay is not None and args.xmap_betay <= 0:
    args.xmap_betay = None
if args.mark_betay is not None and args.mark_betay <= 0:
    args.mark_betay = None
args.mark_points = [tuple(float(v) for v in p.split(",")) for p in args.mark_points]
config_path = args.config.expanduser().resolve()
if not config_path.is_file():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

output_dir = (
    args.output_dir.expanduser().resolve()
    if args.output_dir
    else Path(__file__).resolve().parent / "outputs" / f"peak_lumi_scan_{config_path.stem}"
)
output_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(config_path, output_dir / config_path.name)

# %%
myLumiModel = lumi_class.LumiModel(config_path=str(config_path))

bl_s = args.bunch_length_s if args.bunch_length_s is not None else myLumiModel.init_bunch_length_s
sigma_z = bl_s / 4. * clight
if args.bunches is not None:
    myLumiModel.number_of_bunches_coll_ip15 = args.bunches
elif myLumiModel.number_of_bunches_coll_ip15 is None:
    myLumiModel.number_of_bunches_coll_ip15 = myLumiModel.number_of_bunches
nb = myLumiModel.number_of_bunches_coll_ip15

# cases: name -> intensity, normalized emittance [m], half crab angle [urad] (both beams unless
# crab_b2_urad gives a different one for B2)
cases = {"hllhc": dict(intensity=args.intensity if args.intensity is not None else myLumiModel.init_intensity,
                       emit=(args.emittance_um if args.emittance_um is not None
                             else myLumiModel.init_normalized_emittance) * 1e-6,
                       crab_urad=args.crab_urad)}
if not args.no_ref:
    cases["lhc_ref"] = dict(intensity=args.ref_intensity, emit=args.ref_emittance_um * 1e-6,
                            crab_urad=args.ref_crab_urad)
# HL-LHC without crab cavities (sigma*, beta* and map) and with only B1 crabbed (map only)
cases["hllhc_nocrab"] = dict(cases["hllhc"], crab_urad=0.0)
cases["hllhc_crab_b1"] = dict(cases["hllhc"], crab_b2_urad=0.0)
sigma_cases = [n for n in ("hllhc", "hllhc_nocrab", "lhc_ref") if n in cases]
map_cases = ["hllhc", "hllhc_nocrab", "hllhc_crab_b1"]


def crab_b1_b2_urad(c):
    return c["crab_urad"], c.get("crab_b2_urad", c["crab_urad"])


def crab_text(c):
    b1, b2 = crab_b1_b2_urad(c)
    if b1 == b2:
        return rf"crab$_x$={b1:g} $\mu$rad"
    return rf"crab$_x$ B1={b1:g}, B2={b2:g} $\mu$rad"

# state of the case being scanned, set before forking the workers
case = None


def set_case(name):
    global case
    case = dict(cases[name], name=name)
    case["emit_geom"] = case["emit"] / myLumiModel.gamma
    myLumiModel.current_emitx_B1 = myLumiModel.current_emitx_B2 = case["emit"]
    myLumiModel.current_emity_B1 = myLumiModel.current_emity_B2 = case["emit"]
    myLumiModel.current_bl_s_B1 = myLumiModel.current_bl_s_B2 = bl_s
    myLumiModel.current_inten_B1 = myLumiModel.current_inten_B2 = case["intensity"]


def half_xing_rad(sigmax):
    if args.xing_urad is not None:
        return args.xing_urad * 1e-6
    return 0.5 * args.lr_sep_sigma * case["emit_geom"] / sigmax


def half_crab_rad(half_xing, c=None):
    """(B1, B2) half crab angles [rad] for case `c` (default: the case being scanned)."""
    crabs = []
    for crab_urad in crab_b1_b2_urad(case if c is None else c):
        half_crab = crab_urad * 1e-6
        if not args.no_crab_clip and abs(half_crab) > half_xing:
            # do not crab a beam more than the crossing angle (full compensation at most)
            half_crab = np.sign(half_crab) * half_xing
        crabs.append(half_crab)
    return tuple(crabs)


def peak_lumi(point):
    """Peak luminosity at (beta*_x, beta*_y) [m], or (beta*_x, beta*_y, half crossing [rad]) to set the
    angle explicitly instead of from the LR separation."""
    betax, betay = point[:2]
    half_xing = point[2] if len(point) > 2 else half_xing_rad(np.sqrt(betax * case["emit_geom"]))
    # crossing and crab in x; Lumi_integrals takes the half angles
    crab_b1, crab_b2 = half_crab_rad(half_xing)
    # per-beam crab angles through the model's crab fractions: crab = the larger one, scaled per beam
    crab = crab_b1 if abs(crab_b1) >= abs(crab_b2) else crab_b2
    myLumiModel.cc_b1_fraction = crab_b1 / crab if crab else 0.0
    myLumiModel.cc_b2_fraction = crab_b2 / crab if crab else 0.0
    lumi = myLumiModel.Lumi_integrals(nb, betax=betax, betay=betay,
                                      x=0, px=half_xing, y=0, py=0, ccx=crab, ccy=0, ccrf=400e6)
    return float(np.mean(lumi))


xing_info = (f"half xing={args.xing_urad:g} urad" if args.xing_urad is not None
             else f"LR separation={args.lr_sep_sigma:g} sigma")


def run_scan(name, betas, label):
    """Peak luminosity for a list of (beta*_x, beta*_y) [m] of case `name`, optionally with the half
    crossing angle [rad] as a third element."""
    set_case(name)
    print(f"[{name}, {label}] Scanning {len(betas)} points with {args.jobs} processes "
          f"(N={case['intensity']:.3g}, emit={case['emit']*1e6:.3g} um, bl={bl_s*1e9:.4g} ns, nb={nb}, "
          f"{'half xing scanned' if len(betas[0]) > 2 else xing_info}, half crab B1/B2={'/'.join(f'{v:g}' for v in crab_b1_b2_urad(case))} urad"
          f"{'' if args.no_crab_clip or not any(crab_b1_b2_urad(case)) else ' (clipped to half xing)'})")
    start_time = time.time()
    with mp.get_context("fork").Pool(args.jobs) as pool:
        lumis = pool.map(peak_lumi, betas, chunksize=1)
    print(f"Finished in {time.time()-start_time:.0f}s")

    emit_geom = case["emit_geom"]
    explicit_xing = len(betas[0]) > 2
    results = pd.DataFrame([b[:2] for b in betas], columns=["betax_m", "betay_m"])
    results.insert(0, "case", name)
    results["intensity"] = case["intensity"]
    results["emittance_um"] = case["emit"] * 1e6
    results["betax_cm"] = results.pop("betax_m") * 1e2
    results["betay_cm"] = results.pop("betay_m") * 1e2
    results["ratio_xy"] = results["betax_cm"] / results["betay_cm"]
    results["sigmax_um"] = np.sqrt(results["betax_cm"] * 1e-2 * emit_geom) * 1e6
    results["sigmay_um"] = np.sqrt(results["betay_cm"] * 1e-2 * emit_geom) * 1e6
    results["half_xing_urad"] = ([b[2] * 1e6 for b in betas] if explicit_xing
                                 else [half_xing_rad(sx * 1e-6) * 1e6 for sx in results["sigmax_um"]])
    crabs = np.array([half_crab_rad(hx * 1e-6) for hx in results["half_xing_urad"]]) * 1e6
    results["half_crab_b1_urad"], results["half_crab_b2_urad"] = crabs[:, 0], crabs[:, 1]
    # effective compensation of the half crossing angle: average over the two beams
    results["half_crab_urad"] = crabs.mean(axis=1)
    results["lr_sep_sigma"] = 2 * results["half_xing_urad"] * 1e-6 * results["sigmax_um"] * 1e-6 / emit_geom
    results["piwinski"] = results["half_xing_urad"] * sigma_z / results["sigmax_um"]
    # RMS of the per-beam residual tilts (see header)
    residual_b1 = results["half_xing_urad"] - results["half_crab_b1_urad"].abs()
    residual_b2 = results["half_xing_urad"] - results["half_crab_b2_urad"].abs()
    results["piwinski_eff"] = (np.sqrt(0.5 * (residual_b1**2 + residual_b2**2))
                               * sigma_z / results["sigmax_um"])
    results["lumi_m2s"] = lumis
    results["lumi_1e34"] = results["lumi_m2s"] / 1e38
    results["pileup"] = myLumiModel.cmp_pileup(results["lumi_m2s"])
    return results


def piwinski_eff_of_betax_cm(name, betax_cm):
    """Effective Piwinski angle (crossing minus crab) vs beta*_x [cm] for case `name`."""
    emit_geom = cases[name]["emit"] / myLumiModel.gamma
    sigmax = np.sqrt(np.asarray(betax_cm) * 1e-2 * emit_geom)
    half_xing = (np.full_like(sigmax, args.xing_urad * 1e-6) if args.xing_urad is not None
                 else 0.5 * args.lr_sep_sigma * emit_geom / sigmax)
    residual_sq = 0
    for crab_urad in crab_b1_b2_urad(cases[name]):
        crab = abs(crab_urad) * 1e-6
        residual = half_xing - (crab if args.no_crab_clip else np.minimum(crab, half_xing))
        residual_sq = residual_sq + 0.5 * residual**2
    # RMS of the per-beam residual tilts (see header)
    return np.sqrt(residual_sq) * sigma_z / sigmax


def piwinski_ticks(coord, phi, values=(0.25, 0.5, 1, 2, 4, 8, 16)):
    """Tick positions on `coord` where phi_eff crosses round values. phi_eff is 0 (not invertible)
    once the crab fully compensates the crossing, so a single "0" tick marks where that starts."""
    ticks, labels = [], []
    for p in values:
        for i in np.where(np.diff(np.sign(phi - p)) != 0)[0]:
            ticks.append(coord[i] + (p - phi[i]) * (coord[i + 1] - coord[i]) / (phi[i + 1] - phi[i]))
            labels.append(f"{p:g}")
    zero = phi <= 1e-12
    edges = np.where(np.diff(zero.astype(int)) != 0)[0]
    if len(edges):
        ticks.append(coord[edges[0] + 1] if zero[edges[0] + 1] else coord[edges[0]])
        labels.append("0")
    return ticks, labels


def add_piwinski_axis(ax, name, xlim):
    """Top axis with effective Piwinski ticks vs beta*_x [cm]."""
    top = ax.twiny()
    top.set_xlim(xlim)
    b = np.linspace(max(xlim[0], 1e-3), xlim[1], 4000)
    ticks, labels = piwinski_ticks(b, piwinski_eff_of_betax_cm(name, b))
    top.set_xticks(ticks)
    top.set_xticklabels(labels)
    top.tick_params(labelsize=16)
    top.set_xlabel(r"Piwinski angle $\phi_{Piwinski}$ (with crab)" if any(crab_b1_b2_urad(cases[name]))
                   else r"Piwinski angle $\phi_{Piwinski}$", fontsize=22)
    return top


def plot_lines(results, name, xcol, xlabel, curve_col, curve_label, xlim, piwinski_axis=False):
    c = cases[name]
    # luminosity, pile-up and the angles used, in stacked panels sharing the x axis
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 15), gridspec_kw=dict(height_ratios=[2, 2, 1]))
    for i, (value, group) in enumerate(results.groupby(curve_col, sort=False)):
        style = dict(color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                     lw=2, ms=8, label=curve_label(value))
        ax[0].plot(group[xcol], group["lumi_1e34"], **style)
        ax[1].plot(group[xcol], group["pileup"], **style)
    # the angles depend on the x-plane optics only
    angles = results.drop_duplicates(xcol)
    ax[2].plot(angles[xcol], angles["half_xing_urad"], color="k", lw=2, marker="o", ms=8,
               label="half crossing")
    if any(crab_b1_b2_urad(c)):
        ax[2].plot(angles[xcol], np.abs(angles["half_crab_urad"]), color="k", lw=2, ls="--", marker="s",
                   ms=8, label="|half crab|" if len(set(crab_b1_b2_urad(c))) == 1 else "|half crab|, B1/B2 avg")
    ax[0].set_ylabel(r"Peak lumi [$10^{34}$ Hz/cm$^2$]")
    ax[1].set_ylabel("Peak pile-up")
    ax[2].set_ylabel(r"Angle [$\mu$rad]")
    ax[2].set_xlabel(xlabel)
    title_xing = (rf"$\theta_x/2$={args.xing_urad:g} $\mu$rad" if args.xing_urad is not None
                  else rf"$d_{{LR}}$={args.lr_sep_sigma:g}$\sigma$")
    label = CASE_LABELS[name]
    title = (rf"{label}: N={c['intensity']:.3g}, $\epsilon_n$={c['emit']*1e6:g} $\mu$m, {title_xing}, "
             rf"{crab_text(c)}")
    ax[2].set_xlim(xlim)
    if piwinski_axis:
        add_piwinski_axis(ax[0], name, xlim)
        fig.suptitle(title, fontsize=16)
    else:
        ax[0].set_title(title, fontsize=16)
    for a in ax:
        a.grid(alpha=0.3)
    # headroom so the legend sits above the curves
    ax[0].set_ylim(0, 1.3 * results["lumi_1e34"].max())
    ax[1].set_ylim(0, 1.05 * results["pileup"].max())
    ax[0].legend(ncol=3, loc="upper center")
    ax[2].legend()
    return fig, ax


def plot_fixed_betay(results, names, betay_cm):
    """Simplified view: lumi and angles vs beta*_x at fixed beta*_y, one curve per crab scenario."""
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 11), gridspec_kw=dict(height_ratios=[2, 1]))
    for i, name in enumerate(names):
        group = results[results["case"] == name]
        style = dict(color=COLORS[i], marker=MARKERS[i], markevery=5, lw=2, ms=8)
        ax[0].plot(group["betax_cm"], group["lumi_1e34"], label=f"{CASE_LABELS[name]} ({crab_text(cases[name])})",
                   **style)
        # optimum of each scenario, labelled with its value
        best = group.loc[group["lumi_1e34"].idxmax()]
        ax[0].plot(best["betax_cm"], best["lumi_1e34"], marker="*", ms=20, color=COLORS[i], mec="k", ls="none")
        ax[0].annotate(rf"{best['lumi_1e34']:.2f} at $\beta^*_x$={best['betax_cm']:g} cm",
                       (best["betax_cm"], best["lumi_1e34"]), xytext=(0, 14), textcoords="offset points",
                       ha="center", fontsize=13, color="#333333")
    # the crossing does not depend on the crab scenario, and every crabbed beam gets the same (clipped) crab
    angles = results.drop_duplicates("betax_cm")
    ax[1].plot(angles["betax_cm"], angles["half_xing_urad"], color="k", lw=2, label="half crossing (all cases)")
    crabbed = results[results["half_crab_b1_urad"] != 0].drop_duplicates("betax_cm")
    if len(crabbed):
        ax[1].plot(crabbed["betax_cm"], crabbed["half_crab_b1_urad"].abs(), color="#777777", lw=2, ls="--",
                   label="|half crab| of the crabbed beam(s)")
    ax[0].set_ylabel(r"Peak lumi [$10^{34}$ Hz/cm$^2$]")
    ax[1].set_ylabel(r"Angle [$\mu$rad]")
    ax[1].set_xlabel(r"$\beta^*_x$ [cm]")
    title_xing = (rf"$\theta_x/2$={args.xing_urad:g} $\mu$rad" if args.xing_urad is not None
                  else rf"$d_{{LR}}$={args.lr_sep_sigma:g}$\sigma$")
    c = cases[names[0]]
    ax[0].set_title(rf"HL-LHC, $\beta^*_y$ = {betay_cm:g} cm: N={c['intensity']:.3g}, "
                    rf"$\epsilon_n$={c['emit']*1e6:g} $\mu$m, {title_xing}", fontsize=16)
    ax[0].set_ylim(0, 1.12 * results["lumi_1e34"].max())
    ax[1].set_ylim(0, None)
    ax[1].set_xlim(results["betax_cm"].min() - 1, results["betax_cm"].max() + 1)
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="upper right")
    return fig, ax


def plot_map(results, name):
    c = cases[name]
    grid = results.pivot(index="half_xing_urad", columns="ratio_xy", values="lumi_1e34")
    ratios, xings = grid.columns.values, grid.index.values
    fig, ax = plt.subplots(figsize=(14, 9))
    mesh = ax.pcolormesh(ratios, xings, grid.values, cmap=args.map_cmap, shading="gouraud")
    levels = np.arange(1, np.ceil(grid.values.max()) + 1, 1)
    # white contours with a dark halo stay readable on both ends of the colormap
    halo = [pe.withStroke(linewidth=2.5, foreground="k", alpha=0.6)]
    cs = ax.contour(ratios, xings, grid.values, levels=levels, colors="w", linewidths=1.2)
    cs.set_path_effects(halo)
    for t in ax.clabel(cs, fmt="%g", fontsize=14):
        t.set_path_effects(halo)
    b1, b2 = crab_b1_b2_urad(c)
    if b1 == b2 != 0 and not args.no_crab_clip:
        # above this angle the crab no longer compensates the full crossing angle
        ax.axhline(abs(c["crab_urad"]), color="w", ls="--", lw=1.5, path_effects=halo)
        ax.text(ratios[-1], abs(c["crab_urad"]) + 3, "full crab compensation below ", va="bottom", ha="right",
                fontsize=14, color="k", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    # beta*_x [cm] = k / (theta/2 [rad])^2 at constant LR separation
    k = (0.5 * args.lr_sep_sigma)**2 * c["emit"] / myLumiModel.gamma * 1e2
    if args.mark_betay is not None:
        # constant beta*_y: beta*_x = ratio * beta*_y
        r = np.linspace(ratios[0], ratios[-1], 400)
        ax.plot(r, np.sqrt(k / (r * args.mark_betay)) * 1e6, color="w", ls=":", lw=2.5, path_effects=halo,
                label=rf"$\beta^*_y$ = {args.mark_betay:g} cm")
    # stars: white, then black, so they are told apart without relying on color
    for (bx, by), face, edge in zip(args.mark_points, ["w", "k", "#2a78d6", "#eb6834"], ["k", "w", "w", "w"]):
        set_case(name)
        lumi = peak_lumi((bx * 1e-2, by * 1e-2))
        hx = np.sqrt(k / bx) * 1e6
        print(f"[{name}] beta*_x={bx:g} cm, beta*_y={by:g} cm: half xing={hx:.0f} urad, "
              f"lumi={lumi/1e38:.2f}e34, PU={myLumiModel.cmp_pileup(lumi):.0f}")
        ax.plot(bx / by, hx, marker="*", ms=26, mfc=face, mec=edge, mew=1.5, ls="none",
                label=rf"$\beta^*_x/\beta^*_y$ = {bx:g}/{by:g} cm: "
                      rf"{lumi/1e38:.1f}$\times10^{{34}}$, PU {myLumiModel.cmp_pileup(lumi):.0f}")
    if args.mark_betay is not None or args.mark_points:
        ax.legend(loc="lower left", fontsize=14, framealpha=0.85)
    ax.set_xlim(ratios[0], ratios[-1])
    ax.set_ylim(xings[0], xings[-1])
    cbar = fig.colorbar(mesh, ax=ax, pad=0.23)
    cbar.set_label(r"Peak lumi [$10^{34}$ Hz/cm$^2$]")
    ax.set_xlabel(r"$\beta^*_x/\beta^*_y$")
    ax.set_ylabel(r"Half crossing angle [$\mu$rad]")
    # beta*_x that gives this angle at constant LR separation
    right = ax.secondary_yaxis("right", functions=(lambda t: k / np.maximum(t * 1e-6, 1e-12)**2,
                                                   lambda b: np.sqrt(k / np.maximum(b, 1e-12)) * 1e6))
    right.set_ylabel(r"$\beta^*_x$ [cm]")
    # second right axis, offset outwards: effective Piwinski angle (depends on the angle only)
    par = ax.twinx()
    par.spines["right"].set_position(("axes", 1.16))
    par.set_ylim(ax.get_ylim())
    t = np.linspace(*ax.get_ylim(), 4000)
    ticks, labels = piwinski_ticks(t, piwinski_eff_of_betax_cm(name, k / (t * 1e-6)**2))
    par.set_yticks(ticks)
    par.set_yticklabels(labels)
    par.set_ylabel(r"Piwinski angle $\phi_{Piwinski}$ (with crab)" if any(crab_b1_b2_urad(c))
                   else r"Piwinski angle $\phi_{Piwinski}$")
    ax.set_title(rf"{CASE_LABELS[name]}: N={c['intensity']:.3g}, $\epsilon_n$={c['emit']*1e6:g} $\mu$m, "
                 rf"$d_{{LR}}$={args.lr_sep_sigma:g}$\sigma$, {crab_text(c)}", fontsize=16)
    return fig, ax


def plot_xing_map(results, name, betay_cm, optimum=None, vlim=None):
    """Peak lumi vs beta*_x and half crossing angle at fixed beta*_y, with LR separation contours.
    `optimum`: (beta*_x [cm], half crossing [urad], lumi) of the best point at the nominal LR separation.
    `vlim`: (min, max) of the color scale, shared between scenarios."""
    c = cases[name]
    grid = results.pivot(index="half_xing_urad", columns="betax_cm", values="lumi_1e34")
    betax, xings = grid.columns.values, grid.index.values
    fig, ax = plt.subplots(figsize=(13, 10.5), layout="constrained")
    vmin, vmax = vlim if vlim is not None else (None, None)
    mesh = ax.pcolormesh(betax, xings, grid.values, cmap=args.map_cmap, shading="gouraud", vmin=vmin, vmax=vmax)
    halo = [pe.withStroke(linewidth=2.5, foreground="k", alpha=0.6)]
    cs = ax.contour(betax, xings, grid.values, levels=np.arange(1, np.ceil(grid.values.max()) + 1, 1),
                    colors="w", linewidths=1.2)
    cs.set_path_effects(halo)
    for t in ax.clabel(cs, fmt="%g", fontsize=14):
        t.set_path_effects(halo)
    # LR separation d = theta_full * sqrt(beta*_x / eps_geom), see header
    b = np.linspace(betax[0], betax[-1], 400)
    t = np.linspace(xings[0], xings[-1], 400)
    B, T = np.meshgrid(b, t)
    dsep = 2 * T * 1e-6 * np.sqrt(B * 1e-2 / (c["emit"] / myLumiModel.gamma))
    ds = ax.contour(B, T, dsep, levels=sorted(args.xmap_dsep), colors="#7fd3ff", linewidths=2, linestyles="--")
    ds.set_path_effects(halo)
    for t in ax.clabel(ds, fmt=lambda v: rf"$d_{{LR}}$={v:g}$\sigma$", fontsize=14):
        t.set_path_effects(halo)
    # effective Piwinski angle, RMS of the per-beam residual tilts with the crabs clipped as in the scan
    sigmax = np.sqrt(B * 1e-2 * c["emit"] / myLumiModel.gamma)
    residual_sq = 0
    for crab_urad in crab_b1_b2_urad(c):
        crab = abs(crab_urad) * 1e-6
        residual = T * 1e-6 - (crab if args.no_crab_clip else np.minimum(crab, T * 1e-6))
        residual_sq = residual_sq + 0.5 * residual**2
    phi = np.sqrt(residual_sq) * sigma_z / sigmax
    has_crab = any(crab_b1_b2_urad(c))
    phi_name = r"\phi_{Piwinski}"
    if args.xmap_piwinski:
        pw = ax.contour(B, T, phi, levels=sorted(args.xmap_piwinski), colors="#b8f55a", linewidths=2,
                        linestyles="-.")
        pw.set_path_effects(halo)
        for t in ax.clabel(pw, fmt=lambda v: rf"${phi_name}$={v:g}", fontsize=14):
            t.set_path_effects(halo)
    b1, b2 = crab_b1_b2_urad(c)
    if any((b1, b2)) and not args.no_crab_clip:
        # below this angle the crab fully compensates the crossing
        crab = max(abs(b1), abs(b2))
        ax.axhline(crab, color="w", ls=":", lw=2, path_effects=halo)
        text = "full crab compensation below " if b1 == b2 else "crabbed beam fully compensated below "
        ax.text(betax[-1], crab + 3, text, va="bottom", ha="right", fontsize=14,
                color="k", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    if optimum is not None:
        bx, hx, lumi = optimum
        ax.plot(bx, hx, marker="*", ms=26, mfc="w", mec="k", mew=1.5, ls="none",
                label=rf"best at $d_{{LR}}$={args.lr_sep_sigma:g}$\sigma$: $\beta^*_x$={bx:g} cm, "
                      rf"$\theta/2$={hx:.0f} $\mu$rad, {lumi:.2f}$\times10^{{34}}$")
    # legend entries for the contour families, so they are not told apart by color alone
    ax.plot([], [], color="w", lw=1.2, path_effects=halo, label=r"peak lumi [$10^{34}$ Hz/cm$^2$]")
    ax.plot([], [], color="#7fd3ff", lw=2, ls="--", path_effects=halo, label="LR separation")
    if args.xmap_piwinski:
        ax.plot([], [], color="#b8f55a", lw=2, ls="-.", path_effects=halo,
                label=(rf"Piwinski angle ${phi_name}$ (with crab)" if has_crab else rf"Piwinski angle ${phi_name}$"))
    # below the axes, so it covers none of the contours
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=14, frameon=False)
    ax.set_xlim(betax[0], betax[-1])
    ax.set_ylim(xings[0], xings[-1])
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"Peak lumi [$10^{34}$ Hz/cm$^2$]")
    ax.set_xlabel(r"$\beta^*_x$ [cm]")
    ax.set_ylabel(r"Half crossing angle [$\mu$rad]")
    ax.set_title(rf"{CASE_LABELS[name]}, $\beta^*_y$ = {betay_cm:g} cm: N={c['intensity']:.3g}, "
                 rf"$\epsilon_n$={c['emit']*1e6:g} $\mu$m, {crab_text(c)}", fontsize=16)
    return fig, ax


def frange(start, stop, step):
    return np.round(np.arange(start, stop + 0.5 * step, step), 6)


# %%
# sigma* scans, all cases
sigmax_um = frange(args.sigmax_min, args.sigmax_max, args.sigmax_step)
sigma_points = [(sx, sy) for sy in args.sigmay for sx in sigmax_um]
all_results = []
for name in sigma_cases:
    emit_geom = cases[name]["emit"] / myLumiModel.gamma
    betas = [((sx * 1e-6)**2 / emit_geom, (sy * 1e-6)**2 / emit_geom) for sx, sy in sigma_points]
    results = run_scan(name, betas, "sigma* scan")
    # keep the requested sigma* values exactly (no round-off from the beta* conversion)
    results["sigmax_um"], results["sigmay_um"] = zip(*sigma_points)
    print(results.pivot(index="sigmax_um", columns="sigmay_um", values="lumi_1e34").round(2).to_string())
    all_results.append(results)
    fig, ax = plot_lines(results, name, "sigmax_um", r"$\sigma^*_x$ [$\mu$m]",
                         "sigmay_um", lambda v: rf"$\sigma^*_y$ = {v:g} $\mu$m", xlim=(0, max(30, args.sigmax_max)))
    fig.savefig(output_dir / ("peak_lumi_vs_sigmax.png" if name == "hllhc" else f"peak_lumi_vs_sigmax_{name}.png"))
pd.concat(all_results, ignore_index=True).to_csv(output_dir / "scan.csv", index=False)

# %%
# beta* scan, HL-LHC with and without crab cavities: one curve per beta*_x/beta*_y,
# effective Piwinski angle on the top axis
betax_cm = frange(args.betax_min, args.betax_max, args.betax_step)
beta_points = [(bx, bx / r) for r in args.ratios for bx in betax_cm]
beta_results = []
for name in ["hllhc", "hllhc_nocrab"]:
    results_beta = run_scan(name, [(bx * 1e-2, by * 1e-2) for bx, by in beta_points], "beta* scan")
    results_beta["betax_cm"], results_beta["betay_cm"] = zip(*beta_points)
    results_beta["ratio_xy"] = np.repeat(args.ratios, len(betax_cm))
    beta_results.append(results_beta)
    print(results_beta.pivot(index="betax_cm", columns="ratio_xy", values="lumi_1e34").round(2).to_string())
    print(results_beta.drop_duplicates("betax_cm")[["betax_cm", "sigmax_um", "half_xing_urad", "half_crab_urad",
                                                    "piwinski", "piwinski_eff"]].round(2).to_string(index=False))
    fig, ax = plot_lines(results_beta, name, "betax_cm", r"$\beta^*_x$ [cm]",
                         "ratio_xy", lambda v: rf"$\beta^*_x/\beta^*_y$ = {v:g}",
                         xlim=(args.betax_min - args.betax_step / 2, args.betax_max + args.betax_step / 2),
                         piwinski_axis=True)
    fig.savefig(output_dir / ("peak_lumi_vs_betax.png" if name == "hllhc" else f"peak_lumi_vs_betax_{name}.png"))
pd.concat(beta_results, ignore_index=True).to_csv(output_dir / "scan_beta.csv", index=False)

# %%
# beta*_x scan at fixed beta*_y, HL-LHC: both beams crabbed, B1 only, no crab
results_fixed = None
fixed_cases = ["hllhc", "hllhc_crab_b1", "hllhc_nocrab"]
if args.fixed_betay is not None:
    fixed_betax_cm = frange(args.fixed_betax_min, args.fixed_betax_max, args.fixed_betax_step)
    fixed = []
    for name in fixed_cases:
        res = run_scan(name, [(bx * 1e-2, args.fixed_betay * 1e-2) for bx in fixed_betax_cm],
                       f"beta*_y={args.fixed_betay:g} cm")
        res["betax_cm"], res["betay_cm"] = fixed_betax_cm, args.fixed_betay
        best = res.loc[res["lumi_1e34"].idxmax()]
        print(f"  optimum: beta*_x={best['betax_cm']:g} cm, L={best['lumi_1e34']:.3f}e34, "
              f"PU={best['pileup']:.0f}, half xing={best['half_xing_urad']:.0f} urad, "
              f"half crab B1/B2={best['half_crab_b1_urad']:.0f}/{best['half_crab_b2_urad']:.0f} urad")
        fixed.append(res)
    results_fixed = pd.concat(fixed, ignore_index=True)
    results_fixed.to_csv(output_dir / "scan_fixed_betay.csv", index=False)
    fig, ax = plot_fixed_betay(results_fixed, fixed_cases, args.fixed_betay)
    fig.savefig(output_dir / f"peak_lumi_vs_betax_betay{args.fixed_betay:g}cm.png")

# %%
# map at fixed beta*_y, HL-LHC: beta*_x and half crossing angle scanned independently, LR separation
# shown as contours instead of being imposed
results_xmap = None
if args.xmap_betay is not None:
    xmap_betax = frange(args.xmap_betax_min, args.xmap_betax_max, args.xmap_betax_step)
    xmap_xings = frange(args.xmap_xing_min, args.xmap_xing_max, args.xmap_xing_step)
    xmap_points = [(bx * 1e-2, args.xmap_betay * 1e-2, hx * 1e-6) for hx in xmap_xings for bx in xmap_betax]
    xmaps = []
    for name in fixed_cases:
        res = run_scan(name, xmap_points, f"beta*_x-crossing map, beta*_y={args.xmap_betay:g} cm")
        # exact grid coordinates for the pivot
        res["betax_cm"] = np.tile(xmap_betax, len(xmap_xings))
        res["half_xing_urad"] = np.repeat(xmap_xings, len(xmap_betax))
        xmaps.append(res)
    vlim = (min(r["lumi_1e34"].min() for r in xmaps), max(r["lumi_1e34"].max() for r in xmaps))
    for name, res in zip(fixed_cases, xmaps):
        optimum = None
        if results_fixed is not None and args.xing_urad is None and args.fixed_betay == args.xmap_betay:
            best = results_fixed[results_fixed["case"] == name]
            best = best.loc[best["lumi_1e34"].idxmax()]
            optimum = (best["betax_cm"], best["half_xing_urad"], best["lumi_1e34"])
        fig, ax = plot_xing_map(res, name, args.xmap_betay, optimum, vlim)
        suffix = "" if name == "hllhc" else f"_{name}"
        fig.savefig(output_dir / f"peak_lumi_map_betax_xing_betay{args.xmap_betay:g}cm{suffix}.png")
    results_xmap = pd.concat(xmaps, ignore_index=True)
    results_xmap.to_csv(output_dir / "scan_map_betax_xing.csv", index=False)

# %%
# luminosity map, HL-LHC: beta*_x/beta*_y vs half crossing angle. At constant LR separation
# the angle fixes beta*_x = (d/2)^2 eps_geom / (theta/2)^2, so the map covers beta*_x as well.
results_map = None
if args.xing_urad is None:
    emit_geom = cases["hllhc"]["emit"] / myLumiModel.gamma
    map_ratios = frange(args.map_ratio_min, args.map_ratio_max, args.map_ratio_step)
    map_xings = frange(args.map_xing_min, args.map_xing_max, args.map_xing_step)
    map_points = []
    for hx in map_xings:
        bx = (0.5 * args.lr_sep_sigma)**2 * emit_geom / (hx * 1e-6)**2
        map_points += [(bx, bx / r) for r in map_ratios]
    maps = []
    for name in map_cases:
        res = run_scan(name, map_points, "map")
        # exact grid coordinates for the pivot
        res["ratio_xy"] = np.tile(map_ratios, len(map_xings))
        res["half_xing_urad"] = np.repeat(map_xings, len(map_ratios))
        maps.append(res)
        fig, ax = plot_map(res, name)
        fig.savefig(output_dir / ("peak_lumi_map.png" if name == "hllhc" else f"peak_lumi_map_{name}.png"))
    results_map = pd.concat(maps, ignore_index=True)
    results_map.to_csv(output_dir / "scan_map.csv", index=False)
else:
    print("Skipping the luminosity map: it needs the constant LR separation mode (no --xing-urad)")

with open(output_dir / "scan_parameters.json", "w") as f:
    json.dump(dict(config=config_path.name, bunch_length_s=bl_s, sigma_z_m=sigma_z, colliding_bunches=int(nb),
                   energy_GeV=myLumiModel.En,
                   cases={k: dict(intensity=v["intensity"], normalized_emittance_um=v["emit"] * 1e6,
                                  half_crab_x_b1_urad=crab_b1_b2_urad(v)[0],
                                  half_crab_x_b2_urad=crab_b1_b2_urad(v)[1]) for k, v in cases.items()},
                   lr_sep_sigma=None if args.xing_urad is not None else args.lr_sep_sigma,
                   half_xing_x_urad=args.xing_urad, crab_clipped_to_xing=not args.no_crab_clip,
                   sigma_pileup_m2=myLumiModel.sigmaBOff_m2, sigmay_um=args.sigmay,
                   sigmax_um=sigmax_um.tolist(), ratios_xy=args.ratios, betax_cm=betax_cm.tolist(),
                   fixed_betay_cm=args.fixed_betay,
                   fixed_betax_cm=None if results_fixed is None else fixed_betax_cm.tolist(),
                   xmap_betay_cm=args.xmap_betay,
                   xmap_betax_cm=None if results_xmap is None else xmap_betax.tolist(),
                   xmap_half_xing_urad=None if results_xmap is None else xmap_xings.tolist(),
                   map_ratios_xy=None if results_map is None else map_ratios.tolist(),
                   map_half_xing_urad=None if results_map is None else map_xings.tolist()), f, indent=2)
print(f"Results saved in {output_dir}")
# %%
