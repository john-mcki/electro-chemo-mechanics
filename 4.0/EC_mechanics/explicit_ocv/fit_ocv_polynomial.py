"""
Fit a polynomial OCV curve to NMC SoC-OCV data and plot the result.

Fit: degree-21 polynomial in Chebyshev-normalized variable
     t = 2*(c - 0.30)/0.69 - 1,  c in [0.30, 0.99]

Coefficients are written to stdout and match the ocv_q* values in parameters.prm.
"""

import os
import sys
import numpy as np
import matplotlib
if "--no-show" in sys.argv or not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C

DATA_FILE   = os.path.expanduser("./SoC_OCV_NMC.txt")
SOC_MIN     = 0.30
SOC_MAX     = 0.99
TARGET_ERR  = 5.0   # mV
DEG_RANGE   = range(6, 35)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = np.loadtxt(DATA_FILE, comments="%")
soc_all, ocv_all = data[:, 0], data[:, 1]

mask = (soc_all >= SOC_MIN) & (soc_all <= SOC_MAX)
soc, ocv = soc_all[mask], ocv_all[mask]

# Map SoC to Chebyshev domain [-1, 1]
def to_norm(c):
    return 2.0 * (c - SOC_MIN) / (SOC_MAX - SOC_MIN) - 1.0

soc_norm = to_norm(soc)

# ---------------------------------------------------------------------------
# Sweep degrees to find minimum satisfying target error
# ---------------------------------------------------------------------------
print(f"{'deg':>4}  {'max_err_mV':>12}")
print("-" * 20)
chosen_deg = None
for deg in DEG_RANGE:
    cf = C.chebfit(soc_norm, ocv, deg)
    err_mv = np.max(np.abs(C.chebval(soc_norm, cf) - ocv)) * 1000
    print(f"{deg:>4}  {err_mv:>12.2f}")
    if err_mv < TARGET_ERR and chosen_deg is None:
        chosen_deg = deg

print()
if chosen_deg is None:
    raise RuntimeError(f"No degree in {DEG_RANGE} achieved < {TARGET_ERR} mV.")

print(f"Chosen degree: {chosen_deg}")

# ---------------------------------------------------------------------------
# Final fit at chosen degree
# ---------------------------------------------------------------------------
cheb_coeffs = C.chebfit(soc_norm, ocv, chosen_deg)
ocv_fit     = C.chebval(soc_norm, cheb_coeffs)
residuals   = ocv_fit - ocv

# Convert to standard polynomial (ascending powers in t)
poly_in_t = C.cheb2poly(cheb_coeffs)

print(f"\nCoefficients q0..q{chosen_deg} (poly in t = 2*(c-{SOC_MIN})/{SOC_MAX-SOC_MIN:.2f}-1, ascending):")
for i, coeff in enumerate(poly_in_t):
    print(f"  ocv_q{i:<2d} = {coeff:+.15e}")

print(f"\nMax |residual|: {np.max(np.abs(residuals))*1000:.3f} mV")
print(f"RMS residual:   {np.sqrt(np.mean(residuals**2))*1000:.3f} mV")

# ---------------------------------------------------------------------------
# Dense evaluation for smooth plot curve
# ---------------------------------------------------------------------------
soc_dense      = np.linspace(SOC_MIN, SOC_MAX, 500)
soc_dense_norm = to_norm(soc_dense)
ocv_dense_fit  = C.chebval(soc_dense_norm, cheb_coeffs)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax_main, ax_res) = plt.subplots(
    2, 1, figsize=(7, 6),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)

ax_main.plot(soc_all, ocv_all, ".", color="steelblue", ms=2, alpha=0.5,
             label="NMC data (all)")
ax_main.plot(soc_dense, ocv_dense_fit, "-", color="tomato", lw=2,
             label=f"Chebyshev poly (deg {chosen_deg})")
ax_main.axvline(SOC_MIN, ls="--", lw=0.8, color="gray")
ax_main.axvline(SOC_MAX, ls="--", lw=0.8, color="gray", label=f"fit domain [{SOC_MIN}, {SOC_MAX}]")
ax_main.set_ylabel("OCV (V)")
ax_main.legend(fontsize=8)
ax_main.set_title("NMC OCV polynomial fit")

ax_res.plot(soc, residuals * 1000, ".", color="darkorange", ms=2, alpha=0.6)
ax_res.axhline(0, color="k", lw=0.8)
ax_res.axhline( TARGET_ERR, ls="--", lw=0.8, color="gray", label=f"±{TARGET_ERR} mV")
ax_res.axhline(-TARGET_ERR, ls="--", lw=0.8, color="gray")
ax_res.set_xlabel("SoC")
ax_res.set_ylabel("Residual (mV)")
ax_res.legend(fontsize=8)

plt.tight_layout()
plt.savefig("ocv_fit.png", dpi=150)
plt.show()
print("\nPlot saved to ocv_fit.png")
