"""The chain constants in toggle_chaos.h must stay out of the periodic windows.

x <- x*x + c is chaotic for most c just past -1.4012, but the parameter interval
is full of periodic windows, where the orbit falls onto a short stable cycle and
a flipped bit decays back onto it instead of spreading. c = -1.4 is one: 72% of
random single-bit flips were forgotten there, against about 10% (ordinary
rounding of the lowest mantissa bits) in a chaotic region. A furnace on such a
constant keeps toggling, so a power profile cannot see the problem, but --verify
loses most of its power to detect a fault in that chain.

The check emulates fmaf in float32 on the CPU, with no GPU and no compiler.
"""
import re
from pathlib import Path

import numpy as np

HEADER = Path("kernels/common/toggle_chaos.h").read_text(encoding="utf-8")
CHAINS = 8


def constants(pattern):
    match = re.search(pattern, HEADER)
    assert match, f"toggle_chaos.h no longer defines the constant ({pattern})"
    start, step = match.group(1), match.group(2)
    return float(start), float(step)


def float_chain_constants():
    start, step = constants(
        r"#define PANTHEON_CHAOS_CONST\(k\)\s*\(\s*(-?[0-9.]+)f\s*-\s*([0-9.]+)f")
    a, b = np.float32(start), np.float32(step)
    return [a - b * np.float32(k) for k in range(CHAINS)]


def test_fp64_constants_match_the_fp32_ones():
    fp32 = constants(
        r"#define PANTHEON_CHAOS_CONST\(k\)\s*\(\s*(-?[0-9.]+)f\s*-\s*([0-9.]+)f")
    fp64 = constants(
        r"#define PANTHEON_CHAOS_CONST_D\(k\)\s*\(\s*(-?[0-9.]+)\s*-\s*([0-9.]+)\s*\*")
    assert fp32 == fp64, "the FP64 furnace must use the same constants as the FP32 ones"


def erased_fraction(cs, seeds, trials=240, horizon=4000):
    """Share of random single-bit flips (any mantissa bit) the orbit forgets."""
    rng = np.random.default_rng(1)
    c64 = np.array(cs, dtype=np.float64)
    steps = 6000 + horizon + 2
    base = np.empty((steps, len(cs)), np.float32)
    x = np.array(seeds, dtype=np.float32)
    for i in range(steps):
        base[i] = x
        x = (x.astype(np.float64) ** 2 + c64).astype(np.float32)
    start = rng.integers(2000, 6000, (len(cs), trials))
    bit = rng.integers(0, 23, (len(cs), trials)).astype(np.uint32)
    cols = np.arange(len(cs))[:, None]
    y = (base[start, cols].view(np.uint32) ^ (np.uint32(1) << bit)).view(np.float32).copy()
    erased = np.zeros((len(cs), trials), bool)
    for i in range(1, horizon + 1):
        y = (y.astype(np.float64) ** 2 + c64[:, None]).astype(np.float32)
        erased |= (y == base[start + i, cols])
    return erased.mean(axis=1)


def test_no_chain_sits_in_a_periodic_window():
    cs = float_chain_constants()
    seeds = [np.float32(0.1) + np.float32(0.01) * np.float32(k) for k in range(CHAINS)]
    with np.errstate(over="ignore", invalid="ignore"):
        fraction = erased_fraction(cs, seeds)
    report = ", ".join(f"{float(c):.4f}: {f * 100:.0f}%" for c, f in zip(cs, fraction))
    # Chaotic constants forget 5 to 20% of flips (the lowest bits); a periodic
    # window forgets 40 to 78%.
    assert fraction.max() < 0.30, f"a chain constant is in a periodic window: {report}"


def test_the_old_chain_zero_would_have_failed():
    """Guard the guard: c = -1.4 has to read as a periodic window."""
    seeds = [np.float32(0.1)]
    with np.errstate(over="ignore", invalid="ignore"):
        fraction = erased_fraction([np.float32(-1.4)], seeds)
    assert fraction[0] > 0.5


def test_orbits_stay_bounded():
    cs = float_chain_constants()
    x = np.array([0.1 + 0.01 * k for k in range(CHAINS)], dtype=np.float32)
    c64 = np.array(cs, dtype=np.float64)
    worst = 0.0
    for _ in range(20000):
        x = (x.astype(np.float64) ** 2 + c64).astype(np.float32)
        worst = max(worst, float(np.abs(x).max()))
    beta = (1 + np.sqrt(1 - 4 * c64.min())) / 2
    assert np.isfinite(x).all() and worst <= beta
