//! Attempt to fit 2024/504 into 2023/1705 without binary tower field features.

use crate::basefold::code::RandomFoldableCode;
use p3::{
    field::{AddtiveNtt, BinaryField, ExtensionField, Field},
    matrix::{bitrev::BitReversableMatrix, dense::RowMajorMatrix, Matrix},
    util::{bit_rev, par_bit_rev},
};
use util::{par_zip, rayon::prelude::*};

#[derive(Clone, Debug)]
pub struct BinaryReedSolomonCode<F> {
    lambda: usize,
    log2_c: usize,
    d: usize,
    ntt: AddtiveNtt<F>,
}

impl<F: BinaryField> BinaryReedSolomonCode<F> {
    pub fn new(lambda: usize, log2_c: usize, d: usize) -> Self {
        Self {
            lambda,
            log2_c,
            d,
            ntt: AddtiveNtt::new(log2_c + d),
        }
    }

    fn twiddle_bo(&self, i: usize) -> impl Iterator<Item = &F> {
        bit_rev(&self.ntt.twiddles()[self.log2_c() + i])
    }

    fn par_twiddle_bo(&self, i: usize) -> impl IndexedParallelIterator<Item = &F> {
        par_bit_rev(&self.ntt.twiddles()[self.log2_c() + i])
    }
}

impl<F: BinaryField> RandomFoldableCode<F> for BinaryReedSolomonCode<F> {
    fn lambda(&self) -> usize {
        self.lambda
    }

    fn log2_c(&self) -> usize {
        self.log2_c
    }

    fn log2_k_0(&self) -> usize {
        0
    }

    fn d(&self) -> usize {
        self.d
    }

    // Formula 8 of 2024/504.
    fn num_queries(&self) -> usize {
        let delta = (1.0 + 1.0 / self.c() as f64) / 2.0;
        (self.lambda() as f64 / -delta.log2()).ceil() as usize
    }

    fn encode0<E: ExtensionField<F>>(&self, m: &[E]) -> Vec<E> {
        debug_assert_eq!(m.len(), 1);

        vec![m[0]; self.c()]
    }

    fn encode_batch(&self, m: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        debug_assert_eq!(m.height(), self.k_d());

        let mut m = m.bit_reverse_rows().to_row_major_matrix();
        m.pad_to_height(self.n_d(), F::ZERO);
        let len = m.width() * self.k_d();
        (1..self.c()).for_each(|i| m.values.copy_within(0..len, i * len));
        par_zip!(
            self.ntt.par_cosets(self.log2_c()),
            m.par_row_chunks_exact_mut(self.k_d())
        )
        .for_each(|(coset_ntt, chunk)| coset_ntt.forward_batch_inplace(chunk));

        // Hacky way to turn even-odd to left-right folding
        m.bit_reverse_rows().to_row_major_matrix()
    }

    fn t_inv_halves(&self, _: usize) -> &[F] {
        unreachable!()
    }

    fn fold<E: ExtensionField<F>>(&self, i: usize, w: &mut Vec<E>, r_i: E) {
        debug_assert_eq!(w.len(), 1 << (self.log2_c() + i + 1));

        let mid = w.len() / 2;
        let (lo, hi) = w.split_at_mut(mid);
        par_zip!(self.par_twiddle_bo(i), lo, hi)
            .for_each(|(t_inv_half, lo, hi)| *lo = interpolate(*t_inv_half, *lo, *hi, r_i));
        w.truncate(mid);
    }

    fn interpolate<E: ExtensionField<F>>(&self, i: usize, j: usize, lo: E, hi: E, r_i: E) -> E {
        interpolate(*self.twiddle_bo(i).nth(j).unwrap(), lo, hi, r_i)
    }
}

/// Returns `(addtive_ntt_butterfly^-1)(lo, hi, t) ⊗ (1 - r_i, r_i)`
#[inline]
fn interpolate<F: Field, E: ExtensionField<F>>(twiddle: F, mut lo: E, mut hi: E, r_i: E) -> E {
    hi += lo;
    lo += hi * twiddle;
    lo + (hi - lo) * r_i
}
