use core::fmt::Debug;
use p3::{
    field::{ExtensionField, Field},
    matrix::dense::RowMajorMatrix,
};
use util::izip;

mod generic;
mod reed_solomon;

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

    /// Returns diagonal matrices `(diag(T_0), . . . , diag(T_{d−1}))`.
    fn ts(&self) -> &[Vec<F>];

    /// Returns weights for interpolation.
    fn weights(&self) -> &[Vec<F>];

    /// Returns `w = m * G_0`.
    fn encode0<E: ExtensionField<F>>(&self, m: &[E]) -> Vec<E>;

    /// Returns `w = m * G_d` where `m = interpolate(polys)`.
    fn batch_encode(&self, m: RowMajorMatrix<F>) -> RowMajorMatrix<F>;

    /// Returns `interpolate((diag(T_i)[j], w[j]), (diag(T_i')[j], w[j+n_i])))`
    fn fold<E: ExtensionField<F>>(&self, i: usize, w: &mut Vec<E>, r_i: E) {
        debug_assert_eq!((w.len() / self.n_0()).ilog2() as usize - 1, i);

        let mid = w.len() / 2;
        let (l, r) = w.split_at_mut(mid);
        izip!(l, r, &self.ts()[i], &self.weights()[i])
            .for_each(|(l, r, t, weight)| *l += (r_i - *t) * (*r - *l) * *weight);
        w.truncate(mid);
    }

    /// Returns `interpolate((diag(T_i)[j], l), (diag(T_i')[j], r)))`
    fn interpolate<E: ExtensionField<F>>(&self, i: usize, j: usize, l: E, r: E, r_i: E) -> E {
        let t = self.ts()[i][j];
        let weight = self.weights()[i][j];
        l + (r_i - t) * (r - l) * weight
    }
}
