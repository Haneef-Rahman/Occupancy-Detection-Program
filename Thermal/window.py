#!/usr/bin/env python3
"""
Correct for an IR window in front of the Lepton, without amplifying its noise.

    python window.py --self-test

WHY A CORRECTION IS NEEDED. Poly FIR 200 at 0.2 mm transmits about 67.5% in the
8-14 um band. The v2 model was trained with NO window, on a fixed 15-45 C span,
so an uncorrected enclosure hands the CNN a systematically lower-contrast image
than anything it ever saw.

WHY IT IS NOT A GAIN. The Lepton is radiometric: it reports absolute
temperature, and the 32.5% the window does not transmit is not replaced by
nothing. It is replaced by the window's own emission and its reflection of the
enclosure interior:

    L_measured = tau * L_scene + emissivity * L_window + rho * L_ambient

So the scene is not scaled toward zero, it is COMPRESSED TOWARD THE WINDOW'S
TEMPERATURE. What the window destroys is contrast:

    dT_measured = tau * dT_true

while the absolute level is roughly held up by the enclosure's own glow.
Multiplying the whole frame by 1/tau would therefore lift a 24 C background to
35 C and shift every pixel out of the band the model was trained on. The
correct operation is affine -- expand the deviation, leave the level:

    T_corrected = reference + (T - reference) / tau

with the reference taken as the frame median, which is the ambient estimate the
rest of this project already uses (people are a minority of the frame).

WHY FFC DOES NOT SAVE YOU. The Lepton's shutter sits BEHIND the window, so its
flat-field correction removes sensor non-uniformity and nothing the window or
the enclosure does. Any enclosure glow correction has to be external. That is
what --calibrate produces.

THE COST, AND THE POINT OF THE FILTERING. Expanding contrast by 1/tau expands
NETD with it: the Lepton 3.1R's ~50 mK becomes ~74 mK, which is the graininess
you would see. Two defences, both on by default:

  temporal   a per-pixel exponential average that is MOTION GATED. Static
             pixels average over many frames (noise falls as 1/sqrt(N)); a
             pixel that changes faster than the noise floor is followed
             immediately, so a walking person is not smeared into a comet.
             This is what buys back the SNR -- and then some.

  spatial    a bilateral filter, not a Gaussian. The feature the omega model
             keys on is the head-and-shoulder boundary, and a Gaussian would
             soften exactly that. Bilateral smooths within a region and
             preserves the step across it.

ORDER MATTERS. Flat field, then denoise, then expand. Denoising happens in
native temperature units where the noise magnitude is known (NETD), so the
bilateral range parameter can be set in kelvin rather than guessed. Expanding
last means the filter never has to chase amplified noise.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# TRANSMISSION vs THICKNESS, and why thickness is the only lever.
#
#     tau = (1 - R)^2 * exp(-alpha * t)
#            ~0.92         absorption
#
# Two Fresnel surfaces at n ~ 1.5 cost about 4% each, so ~8% is lost to
# reflection no matter what. Everything else is bulk absorption -- polyethylene
# has C-H bands near 3.4, 6.8 and 13.7 um, and that last one eats into the top
# of the 8-14 um window. Fitting alpha to the quoted tau(0.2 mm) = 0.675 gives
# 1.55 /mm, and therefore:
#
#     thickness      tau     gain 1/tau    NETD after
#     0.100 mm      0.769       1.30          65 mK
#     0.200 mm      0.675       1.48          74 mK
#     0.300 mm      0.592       1.69          84 mK
#     0.500 mm      0.456       2.19         110 mK     <- the only stock
#     0.600 mm      0.400       2.50         125 mK     <- published
#     1.000 mm      0.237       4.22         211 mK
#
# REVISED. An earlier version of this table fitted alpha to the 0.2 mm point
# alone and assumed an ideal 8% Fresnel loss, giving tau(0.5) = 0.424. Fresnel
# Factory also publish ~40% at 0.6 mm, and fitting BOTH points gives
# alpha = 1.308 /mm with a surface factor of 0.877 -- i.e. 12.3% surface loss,
# higher than ideal, which is what a diffusing white pigment and a textured
# finish would do. The two-point fit reproduces both published figures exactly
# and is interpolation rather than extrapolation, so tau(0.5) = 0.456 is the
# number to trust.
#
# Distance from the lens changes NONE of this: neither term contains a
# standoff. Thickness is the whole game, and the 0.5 mm sheet costs more than
# twice the noise of the 0.2 mm one.
#
# CAVEAT ON THE INPUT. 67.5% is quoted as MAX transmission, i.e. the peak of
# the spectral curve. What matters is the average across 8-14 um weighted by
# the detector response, which is necessarily lower. Treat every tau here as
# optimistic until measured against a known source.
#
# At 60 GHz none of this applies: lambda is 5 mm, so even 0.5 mm is lambda/10
# and polyethylene's loss tangent is ~1e-4. The radar does not care.
# ---------------------------------------------------------------------------

TAU_BY_THICKNESS_MM = {0.10: 0.769, 0.20: 0.675, 0.30: 0.592,
                       0.50: 0.456, 0.60: 0.400, 1.00: 0.237}

POLY_FIR200_TAU = 0.675          # 0.2 mm Poly FIR 200, 8-14 um
POLY_FIR200_TAU_0P5 = 0.456      # 0.5 mm, two-point fit
LEPTON_NETD_C = 0.05             # Lepton 3.1R, approx


class WindowCorrection:
    """
    Toggleable. When `enabled` is False every call is a passthrough, so the
    live A/B is exact rather than approximate -- the same code path with the
    correction switched out, not a separate branch that might have drifted.
    """

    def __init__(self, tau=POLY_FIR200_TAU, enabled=False,
                 temporal=True, spatial=True,
                 netd=LEPTON_NETD_C, alpha_static=0.25, motion_k=3.0,
                 bilateral_d=5, sigma_space=2.0, flat=None):
        self.tau = float(tau)
        self.enabled = bool(enabled)
        self.temporal = temporal
        self.spatial = spatial
        self.netd = float(netd)
        self.alpha_static = float(alpha_static)
        self.motion_k = float(motion_k)
        self.bilateral_d = int(bilateral_d)
        self.sigma_space = float(sigma_space)
        self.flat = flat                      # fixed offset field, degrees C
        self._prev = None
        self.n_frames = 0

    # -- the pieces --------------------------------------------------------

    def _flat_field(self, a):
        """
        Subtract the enclosure's own contribution.

        A FIXED field from an explicit calibration, never an adaptive
        background. That distinction is load-bearing: an adaptive background
        was measured on this project to delete 1842 valid labels, because a
        person seated at a desk becomes their own background. A calibration
        captured once against a uniform scene cannot do that.
        """
        if self.flat is None:
            return a
        return a - self.flat

    def _temporal(self, a):
        """
        Per-pixel EMA whose weight depends on how much that pixel moved.

        A flat EMA would smear anyone walking: at 9 fps a 4-frame average
        drags a person across several pixels. Gating on the change relative to
        the noise floor means still pixels get the averaging and moving pixels
        get none, which is where the SNR comes from without the comet trail.
        """
        if self._prev is None:
            self._prev = a.copy()
            return a
        d = np.abs(a - self._prev)
        moving = d > (self.motion_k * self.netd)
        alpha = np.where(moving, 1.0, self.alpha_static).astype(np.float32)
        out = alpha * a + (1.0 - alpha) * self._prev
        self._prev = out
        return out

    def _spatial(self, a):
        if cv2 is None or self.bilateral_d <= 0:
            return a
        # sigmaColor in KELVIN: smooth differences that are plausibly noise,
        # keep differences that are plausibly a person against the room.
        return cv2.bilateralFilter(a.astype(np.float32), self.bilateral_d,
                                   float(2.0 * self.netd), self.sigma_space)

    def _expand(self, a):
        ref = float(np.median(a))
        return ref + (a - ref) / self.tau

    # -- the whole thing ---------------------------------------------------

    def apply(self, data):
        if not self.enabled:
            return data
        a = np.asarray(data, dtype=np.float32)
        a = self._flat_field(a)
        if self.temporal:
            a = self._temporal(a)
        if self.spatial:
            a = self._spatial(a)
        a = self._expand(a)
        self.n_frames += 1
        return a

    def reset(self):
        self._prev = None

    # -- calibration -------------------------------------------------------

    @staticmethod
    def calibrate(frames, blur=9):
        """
        Build the fixed offset field from frames of a UNIFORM scene.

        Point the enclosed sensor at something flat and thermally even -- a
        blank wall works, a hand does not -- and average. What is left after
        removing the mean is the enclosure's contribution: the window's own
        emission, and the narcissus reflection of the detector and inner wall.

        Only the LOW SPATIAL FREQUENCY part is kept. Enclosure glow is broad
        and smooth; sensor noise is not; and a person is not in the reference
        scene at all. Blurring before storing means a stray warm object during
        calibration cannot burn a person-shaped hole into every later frame.
        """
        stack = np.stack([np.asarray(f, np.float32) for f in frames])
        med = np.median(stack, axis=0)
        field = med - float(np.median(med))
        if cv2 is not None and blur >= 3:
            k = blur | 1
            field = cv2.GaussianBlur(field, (k, k), 0)
        return field

    def save(self, path):
        np.savez_compressed(path, flat=self.flat, tau=self.tau,
                            netd=self.netd)

    @staticmethod
    def load(path, **kw):
        d = np.load(path)
        return WindowCorrection(tau=float(d["tau"]), netd=float(d["netd"]),
                                flat=d["flat"], **kw)

    def status(self):
        bits = []
        if self.temporal:
            bits.append("temporal")
        if self.spatial:
            bits.append("spatial")
        if self.flat is not None:
            bits.append("flat")
        return f"1/{self.tau:.3f}={1/self.tau:.2f}x " + ("+".join(bits) or "raw")


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def self_test():
    ok = True
    rng = np.random.default_rng(0)

    def check(label, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {label} {extra}")

    tau = POLY_FIR200_TAU
    amb, body = 24.0, 32.0                    # 8 C of true contrast
    H, W = 120, 160

    def scene(noise=0.0, moved=0):
        a = np.full((H, W), amb, np.float32)
        a[40:90, 60 + moved:90 + moved] = body
        if noise:
            a = a + rng.normal(0, noise, a.shape).astype(np.float32)
        return a

    def through_window(a, t_window=26.0):
        """What the Lepton actually reports: contrast scaled, level held up."""
        return tau * a + (1.0 - tau) * t_window

    print("the window destroys contrast, not level")
    clean = scene()
    seen = through_window(clean)
    check("true contrast", round(float(clean[60, 75] - clean[10, 10]), 2), 8.0)
    check("measured contrast is tau x true",
          abs(float(seen[60, 75] - seen[10, 10]) - tau * 8.0) < 0.01,
          f"{float(seen[60,75]-seen[10,10]):.2f}")
    check("a naive x1.48 gain would wreck the level",
          abs(float((seen * (1 / tau))[10, 10]) - amb) > 5.0,
          f"background becomes {float((seen*(1/tau))[10,10]):.1f} C")

    print("\nthe affine correction restores it")
    wc = WindowCorrection(tau=tau, enabled=True, temporal=False, spatial=False)
    out = wc.apply(seen)
    check("contrast restored",
          abs(float(out[60, 75] - out[10, 10]) - 8.0) < 0.05,
          f"{float(out[60,75]-out[10,10]):.2f} C")
    check("background stays put",
          abs(float(out[10, 10]) - float(np.median(seen))) < 0.6,
          f"{float(out[10,10]):.1f} C")

    print("\ndisabled is an exact passthrough")
    off = WindowCorrection(tau=tau, enabled=False)
    check("byte-identical", np.array_equal(off.apply(seen), seen))

    print("\nnoise: expansion amplifies it, filtering buys it back")
    noisy = [through_window(scene(noise=LEPTON_NETD_C)) for _ in range(24)]
    raw_sd = float(np.std(noisy[-1][0:30, 0:30]))
    plain = WindowCorrection(tau=tau, enabled=True, temporal=False, spatial=False)
    amp_sd = float(np.std(plain.apply(noisy[-1])[0:30, 0:30]))
    check("expansion amplifies noise by ~1/tau",
          abs(amp_sd / raw_sd - 1 / tau) < 0.1, f"{amp_sd/raw_sd:.2f}x")

    full = WindowCorrection(tau=tau, enabled=True)
    for f in noisy:
        res = full.apply(f)
    filt_sd = float(np.std(res[0:30, 0:30]))
    print(f"       raw {raw_sd*1000:.0f} mK -> expanded {amp_sd*1000:.0f} mK "
          f"-> filtered {filt_sd*1000:.0f} mK")
    check("filtered is quieter than the raw measurement",
          filt_sd < raw_sd, f"{filt_sd*1000:.0f} < {raw_sd*1000:.0f} mK")

    print("\nmotion gate: a walking person is not smeared")
    mg = WindowCorrection(tau=tau, enabled=True, spatial=False)
    for k in range(12):
        r = mg.apply(through_window(scene(noise=LEPTON_NETD_C, moved=2 * k)))
    edge = float(r[60, 60 + 22 + 1] - r[60, 60 + 22 - 8])   # across the body edge
    check("edge contrast survives the temporal filter", abs(edge) > 6.0,
          f"{abs(edge):.1f} C step")
    trail = float(r[60, 20] - np.median(r))
    check("no trail left behind at the start position", abs(trail) < 1.0,
          f"{trail:+.2f} C")

    print("\nflat field: enclosure glow removed, people not")
    glow = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    glow += 1.8 * np.exp(-(((xx - 80) ** 2 + (yy - 60) ** 2) / (2 * 55.0 ** 2)))
    cal = [through_window(np.full((H, W), amb, np.float32)) + glow
           + rng.normal(0, LEPTON_NETD_C, (H, W)).astype(np.float32)
           for _ in range(16)]
    field = WindowCorrection.calibrate(cal)
    # Expected value computed FROM the synthetic glow, not guessed: at (5,5)
    # the Gaussian is still 0.43 C, not ~0, so the corner-to-centre difference
    # is 1.37 and not the 1.7 a first reading suggests.
    want = float(glow[60, 80] - glow[5, 5])
    check("glow captured", abs(float(field[60, 80] - field[5, 5]) - want) < 0.15,
          f"{float(field[60,80]-field[5,5]):.2f} C, expected {want:.2f}")
    wf = WindowCorrection(tau=tau, enabled=True, temporal=False, spatial=False,
                          flat=field)
    corrected = wf.apply(through_window(clean) + glow)
    check("body still 8 C above ambient after correction",
          abs(float(corrected[60, 75] - corrected[10, 10]) - 8.0) < 0.4,
          f"{float(corrected[60,75]-corrected[10,10]):.2f} C")
    resid = float(abs(corrected[10, 10] - corrected[110, 150]))
    check("glow no longer tilts the background", resid < 0.5,
          f"{resid:.2f} C corner-to-corner")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--tau", type=float, default=POLY_FIR200_TAU)
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    ap.print_help()


if __name__ == "__main__":
    main()
