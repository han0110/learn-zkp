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

    /// Returns `w = m * G_0`.
    fn encode0<E: ExtensionField<F>>(&self, m: &[E]) -> Vec<E>;

    /// Returns `w = m * G_d`.
    fn encode_batch(&self, m: RowMajorMatrix<F>) -> RowMajorMatrix<F>;

    /// Returns `interpolate((diag(T_i)[j], w[j]), (diag(T_i')[j], w[j+n_i])))`
    fn fold<E: ExtensionField<F>>(&self, i: usize, w: &mut Vec<E>, r_i: E);

    /// Returns `interpolate((diag(T_i)[j], lo), (diag(T_i')[j], hi)))`
    fn interpolate<E: ExtensionField<F>>(&self, i: usize, j: usize, lo: E, hi: E, r_i: E) -> E;
}

fn fold<F: Field, E: ExtensionField<F>>(t_inv_halves: &[F], w: &mut Vec<E>, r_i: E) {
    let mid = w.len() / 2;
    let (a, b) = w.split_at_mut(mid);
    izip!(t_inv_halves, a, b)
        .for_each(|(t_inv_half, a, b)| *a = interpolate(*t_inv_half, *a, *b, r_i));
    w.truncate(mid);
}

#[inline]
fn interpolate<F: Field, E: ExtensionField<F>>(t_inv_half: F, a: E, b: E, r_i: E) -> E {
    let c = (a + b).halve();
    let d = (a - b) * t_inv_half;
    c + (d - c) * r_i
}
