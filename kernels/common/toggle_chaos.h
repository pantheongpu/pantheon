#ifndef PANTHEON_TOGGLE_CHAOS_H
#define PANTHEON_TOGGLE_CHAOS_H

// For the __device__/__host__ qualifiers, which the mock build defines away.
#include "common.h"

// Bounded-chaos state for the arithmetic furnaces.
//
// A power virus has to make its datapath *toggle*, not merely issue: dynamic
// power is switching activity, so a pipe recomputing the same value at full
// issue rate draws a fraction of what real work draws. Two recurrences fail
// that while looking perfectly busy in a profile, and the furnaces in this
// tree used both.
//
//   Saturation. a=fma(a,b,c); b=fma(b,c,a); c=fma(c,a,b) seeded near 1.0
//   reaches +/-inf after eight FP32 FMAs and seven in FP16. Every FMA after
//   that multiplies inf by inf: same instruction rate, not one bit changes.
//
//   Convergence. a=fma(a,b,c) with |b| < 1 and a small c contracts onto a
//   fixed point within a few dozen steps and then recomputes it forever.
//
// Measured over each kernel's real default loop count, the saturating chains
// toggled 0.0 bits of 32 per step and the converging ones 0.0 to 1.2.
//
// Both also make bitwise verification vacuous. A golden pass that saturated
// compares inf against inf, and one that converged compares a constant
// against the same constant; either way the comparison cannot fail, so a test
// that could never detect an SDC reports PASS. That is the same class of
// defect as a self-test that never ran.
//
// x <- x*x + c with c in [-1.435, -1.4] avoids both:
//
//   * Chaotic, so the mantissa churns on every step -- 13.4 bits of 32 -- and
//     a single flipped bit diverges instead of being healed away.
//   * Closed, so it needs no clamp: |x| <= 1.435 gives x*x <= 2.059 and
//     x*x + c in [-1.435, 0.659], back inside the same interval. Escape
//     would need |x| > 1.685, which the invariant forbids. A clamp would be
//     worse than unnecessary -- it is a path that can heal a real fault.
//   * Seeded from constants, not from the thread index, so the workloads that
//     verify against a single consensus value still have one.
//
// The per-chain index k matters as much as the toggling. One chain issues at
// most one FMA per FMA latency -- four cycles on every current part -- so a
// thread running a single chain caps at a quarter of the pipe's issue rate no
// matter how many warps are resident. Several independent chains per thread
// keep it issuing every cycle.

#define PANTHEON_CHAOS_SEED(k)  (0.1f + 0.01f * (float)(k))
#define PANTHEON_CHAOS_CONST(k) (-1.4f - 0.005f * (float)(k))

// FP64 spellings, so the constants are not rounded through a float literal
// before being widened again.
#define PANTHEON_CHAOS_SEED_D(k)  (0.1 + 0.01 * (double)(k))
#define PANTHEON_CHAOS_CONST_D(k) (-1.4 - 0.005 * (double)(k))

// Bounded-chaos state for the transcendental furnaces.
//
// The same trap catches an SFU chain: sin/cos of a small argument composed
// with a contracting update settles onto a short orbit and stops toggling
// (1.2 bits of 32 measured). Amplifying the argument before the sine makes it
// chaotic while every intermediate stays bounded by construction:
//
//   s, t in [-1, 1]            (sine, cosine)
//   e = exp(-|s*t|) in (0, 1]
//   r = rsqrt(|s| + 0.5) in [0.816, 1.415]
//   b = e + r  in (0.816, 2.415]   -- strictly positive, so log(b) is finite
//   a = log(b) + s*t in (-1.204, 1.882]
//
// Measured at 12.6 bits of 32 per step. The macro takes the chain index for
// the same reason the FMA one does: SFU latency is long, so a single chain
// leaves the special function unit mostly idle.
#define PANTHEON_CHAOS_SFU_SEED_A(k) (0.1f + 0.01f * (float)(k))
#define PANTHEON_CHAOS_SFU_SEED_B(k) (1.2f + 0.01f * (float)(k))

// One step of that chain. Uses the fast intrinsics deliberately: sinf/cosf
// and friends are multi-instruction library routines that spend most of their
// cycles in the FMA pipe, so a test built on them measures the wrong unit.
#define PANTHEON_CHAOS_SFU_STEP(a, b)                      \
    do {                                                   \
        float _s = __sinf((a) * 8.0f + (b));               \
        float _t = __cosf((b) * 5.0f - (a));               \
        float _e = __expf(-fabsf(_s * _t));                \
        float _r = rsqrtf(fabsf(_s) + 0.5f);               \
        (b) = _e + _r;                                     \
        (a) = __logf((b)) + _s * _t;                         \
    } while (0)

// Pseudo-random operand payload for the matrix-core furnaces.
//
// A WMMA loop whose fragments come from fill_fragment alone multiplies the
// same two numbers forever. The multiplier array recomputes identical partial
// products every cycle, and a datapath that does not toggle does not draw
// power -- the same defect as a saturated FMA chain, reached a different way.
// It also makes the accumulator grow monotonically until the fp32 increment
// vanishes into rounding, at which point the adder stops changing too.
//
// Rotating the fragments through a few tiles of uncompressible data fixes
// both, and reloading them puts the LSU and the shared-memory banks under the
// traffic a tiled GEMM's operand staging generates. Keep the reload ratio
// near one load per four MMAs: reloading every fragment every group tips the
// kernel into being LSU-bound and costs tensor throughput.
__device__ __host__ __forceinline__ unsigned int pantheon_operand_hash(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Operands in [-1, 1), so the accumulators random-walk around zero rather
// than growing until the increment is lost to rounding.
#define PANTHEON_OPERAND_VALUE(h) \
    ((float)(int)((h) & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f)

#endif // PANTHEON_TOGGLE_CHAOS_H
