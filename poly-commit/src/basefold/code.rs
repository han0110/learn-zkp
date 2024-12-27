use core::fmt::Debug;
use p3::{
    field::{ExtensionField, Field},
    matrix::dense::RowMajorMatrix,
};
use util::izip;

mod binary_reed_solomon;
mod generic;
mod reed_solomon;

pub use binary_reed_solomon::BinaryReedSolomonCode;
pub use generic::GenericRandomFoldableCode;
pub use reed_solomon::ReedSolomonCode;

pub trait RandomFoldableCode<F: Field>: Debug {
    fn lambda(&self) -> usize;

    fn log2_c(&self) -> usize;

    fn c(&self) -> usize {
        1 << self.log2_c()
    }

    fn log2_k_0(&self) -> usize;

    fn k_0(&self) -> usize {
        1 << self.log2_k_0()
    }

    fn d(&self) -> usize;

    fn n_0(&self) -> usize {
        self.c() * self.k_0()
    }

    fn k_d(&self) -> usize {
        self.k_0() << self.d()
    }

    fn log2_n_d(&self) -> usize {
        self.log2_c() + self.log2_k_0() + self.d()
    }

    fn n_d(&self) -> usize {
        self.c() * self.k_d()
    }

    fn n_i(&self, i: usize) -> usize {
        self.n_0() << i
    }

    /// Returns relative minimum distance.
    fn relative_minimum_distance(&self) -> f64;

    /// Returns number of queries needed to reach `lambda`-bits security.
    fn num_queries(&self) -> usize;

    /// Returns `w = m * G_0`.
    fn encode0<E: ExtensionField<F>>(&self, m: &[E]) -> Vec<E>;

    /// Returns `w = m * G_d`.
    fn encode_batch(&self, m: RowMajorMatrix<F>) -> RowMajorMatrix<F>;

    fn encode(&self, data: Vec<F>) -> Vec<F> {
        self.encode_batch(RowMajorMatrix::new_col(data)).values
    }

    /// Returns `(2*T_i)^-1`.
    fn t_inv_halves(&self, i: usize) -> &[F];

    /// Returns `(dit_butterfly^-1)(lo, hi, diag(T_i)[j]) ⊗ (1 - r_i, r_i)`
    fn fold<E: ExtensionField<F>>(&self, i: usize, w: &mut Vec<E>, r_i: E) {
        debug_assert_eq!(w.len(), 1 << (self.log2_c() + self.log2_k_0() + i + 1));

        let mid = w.len() / 2;
        let (lo, hi) = w.split_at_mut(mid);
        izip!(self.t_inv_halves(i), lo, hi)
            .for_each(|(t_inv_half, lo, hi)| *lo = interpolate(*t_inv_half, *lo, *hi, r_i));
        w.truncate(mid);
    }

    /// Returns `(dit_butterfly^-1)(lo, hi, diag(T_i)[j]) ⊗ (1 - r_i, r_i)`
    fn interpolate<E: ExtensionField<F>>(&self, i: usize, j: usize, lo: E, hi: E, r_i: E) -> E {
        interpolate(self.t_inv_halves(i)[j], lo, hi, r_i)
    }
}

/// Returns `(dit_butterfly^-1)(lo, hi, t) ⊗ (1 - r_i, r_i)`
#[inline]
fn interpolate<F: Field, E: ExtensionField<F>>(t_inv_half: F, lo: E, hi: E, r_i: E) -> E {
    let (lo, hi) = ((lo + hi).halve(), (lo - hi) * t_inv_half);
    lo + (hi - lo) * r_i
}
