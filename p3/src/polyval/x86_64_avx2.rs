// Copied and modified from https://github.com/RustCrypto/universal-hashes/blob/master/polyval/src/backend/clmul.rs.
// Copyright (c) 2019-2023 RustCrypto Developers

use crate::polyval::portable;
use core::{arch::x86_64::*, mem::transmute};

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
        let h = transmute::<u128, __m128i>(h);
        let y = transmute::<u128, __m128i>(y);

        let h0 = h;
        let h1 = _mm_shuffle_epi32(h, 0x0E);
        let h2 = _mm_xor_si128(h0, h1);
        let y0 = y;

        // Multiply values partitioned to 64-bit parts
        let y1 = _mm_shuffle_epi32(y, 0x0E);
        let y2 = _mm_xor_si128(y0, y1);
        let t0 = _mm_clmulepi64_si128(y0, h0, 0x00);
        let t1 = _mm_clmulepi64_si128(y, h, 0x11);
        let t2 = _mm_clmulepi64_si128(y2, h2, 0x00);
        let t2 = _mm_xor_si128(t2, _mm_xor_si128(t0, t1));
        let v0 = t0;
        let v1 = _mm_xor_si128(_mm_shuffle_epi32(t0, 0x0E), t2);
        let v2 = _mm_xor_si128(t1, _mm_shuffle_epi32(t2, 0x0E));
        let v3 = _mm_shuffle_epi32(t1, 0x0E);

        // Polynomial reduction
        let v2 = xor5(
            v2,
            v0,
            _mm_srli_epi64(v0, 1),
            _mm_srli_epi64(v0, 2),
            _mm_srli_epi64(v0, 7),
        );

        let v1 = xor4(
            v1,
            _mm_slli_epi64(v0, 63),
            _mm_slli_epi64(v0, 62),
            _mm_slli_epi64(v0, 57),
        );

        let v3 = xor5(
            v3,
            v1,
            _mm_srli_epi64(v1, 1),
            _mm_srli_epi64(v1, 2),
            _mm_srli_epi64(v1, 7),
        );

        let v2 = xor4(
            v2,
            _mm_slli_epi64(v1, 63),
            _mm_slli_epi64(v1, 62),
            _mm_slli_epi64(v1, 57),
        );

        transmute(_mm_unpacklo_epi64(v2, v3))
    }
}

#[inline(always)]
unsafe fn xor4(e1: __m128i, e2: __m128i, e3: __m128i, e4: __m128i) -> __m128i {
    _mm_xor_si128(_mm_xor_si128(e1, e2), _mm_xor_si128(e3, e4))
}

#[inline(always)]
unsafe fn xor5(e1: __m128i, e2: __m128i, e3: __m128i, e4: __m128i, e5: __m128i) -> __m128i {
    _mm_xor_si128(
        e1,
        _mm_xor_si128(_mm_xor_si128(e2, e3), _mm_xor_si128(e4, e5)),
    )
}

#[inline]
pub fn invert_or_zero(value: u128) -> u128 {
    portable::invert_or_zero(value)
}
