// Copied and modified from https://github.com/RustCrypto/universal-hashes/blob/master/polyval/src/backend/pmull.rs.
// Copyright (c) 2019-2023 RustCrypto Developers

use crate::polyval::portable;
use core::{arch::aarch64::*, mem::transmute};

#[inline]
pub fn from_canonical(a: u128) -> u128 {
    montgomery_multiply(a, 0x1e563df92ea7081b4563df92ea7081b5)
}

#[inline]
pub fn to_canonical(a: u128) -> u128 {
    montgomery_multiply(a, 1)
}

#[inline]
pub fn montgomery_multiply(h: u128, y: u128) -> u128 {
    unsafe {
        let h = vreinterpretq_u8_p128(h);
        let y = vreinterpretq_u8_p128(y);

        let (h, m, l) = karatsuba1(h, y);
        let (h, l) = karatsuba2(h, m, l);
        vreinterpretq_p128_u8(mont_reduce(h, l))
    }
}

/// Karatsuba decomposition for `x*y`.
#[inline]
unsafe fn karatsuba1(x: uint8x16_t, y: uint8x16_t) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    // First Karatsuba step: decompose x and y.
    //
    // (x1*y0 + x0*y1) = (x1+x0) * (y1+y0) + (x1*y1) + (x0*y0)
    //        M                                 H         L
    //
    // m = x.hi^x.lo * y.hi^y.lo
    let m = pmull(
        veorq_u8(x, vextq_u8(x, x, 8)), // x.hi^x.lo
        veorq_u8(y, vextq_u8(y, y, 8)), // y.hi^y.lo
    );
    let h = pmull2(x, y); // h = x.hi * y.hi
    let l = pmull(x, y); // l = x.lo * y.lo
    (h, m, l)
}

/// Karatsuba combine.
#[inline]
unsafe fn karatsuba2(h: uint8x16_t, m: uint8x16_t, l: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    // Second Karatsuba step: combine into a 2n-bit product.
    //
    // m0 ^= l0 ^ h0 // = m0^(l0^h0)
    // m1 ^= l1 ^ h1 // = m1^(l1^h1)
    // l1 ^= m0      // = l1^(m0^l0^h0)
    // h0 ^= l0 ^ m1 // = h0^(l0^m1^l1^h1)
    // h1 ^= l1      // = h1^(l1^m0^l0^h0)
    let t = {
        //   {m0, m1} ^ {l1, h0}
        // = {m0^l1, m1^h0}
        let t0 = veorq_u8(m, vextq_u8(l, h, 8));

        //   {h0, h1} ^ {l0, l1}
        // = {h0^l0, h1^l1}
        let t1 = veorq_u8(h, l);

        //   {m0^l1, m1^h0} ^ {h0^l0, h1^l1}
        // = {m0^l1^h0^l0, m1^h0^h1^l1}
        veorq_u8(t0, t1)
    };

    // {m0^l1^h0^l0, l0}
    let x01 = vextq_u8(
        vextq_u8(l, l, 8), // {l1, l0}
        t,
        8,
    );

    // {h1, m1^h0^h1^l1}
    let x23 = vextq_u8(
        t,
        vextq_u8(h, h, 8), // {h1, h0}
        8,
    );

    (x23, x01)
}

#[inline]
unsafe fn mont_reduce(x23: uint8x16_t, x01: uint8x16_t) -> uint8x16_t {
    const POLY: u128 = (1 << 127) | (1 << 126) | (1 << 121) | (1 << 63) | (1 << 62) | (1 << 57);
    // Perform the Montgomery reduction over the 256-bit X.
    //    [A1:A0] = X0 • poly
    //    [B1:B0] = [X0 ⊕ A1 : X1 ⊕ A0]
    //    [C1:C0] = B0 • poly
    //    [D1:D0] = [B0 ⊕ C1 : B1 ⊕ C0]
    // Output: [D1 ⊕ X3 : D0 ⊕ X2]
    let poly = vreinterpretq_u8_p128(POLY);
    let a = pmull(x01, poly);
    let b = veorq_u8(x01, vextq_u8(a, a, 8));
    let c = pmull2(b, poly);
    veorq_u8(x23, veorq_u8(c, b))
}

/// Multiplies the low bits in `a` and `b`.
#[inline]
unsafe fn pmull(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    transmute(vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 0),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 0),
    ))
}

/// Multiplies the high bits in `a` and `b`.
#[inline]
unsafe fn pmull2(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    transmute(vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 1),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 1),
    ))
}

#[inline]
pub fn invert_or_zero(value: u128) -> u128 {
    portable::invert_or_zero(value)
}
