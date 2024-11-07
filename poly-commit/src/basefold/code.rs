use core::fmt::Debug;
use p3::{
    field::{batch_multiplicative_inverse, ExtensionField, Field, FieldSlice, FromUniformBytes},
    matrix::{
        dense::{RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut},
        Matrix,
    },
    poly::multilinear::batch_interpolate,
};
use rand::RngCore;
use util::{izip, izip_par, rayon::prelude::*, Itertools};

pub trait RandomFoldableCode<F: Field>: Debug + Send + Sync {
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

    /// Returns diagonal matrices `(diag(T_0), . . . , diag(T_{d−1}))`.
    fn ts(&self) -> &[Vec<F>];

    /// Returns `w = m * G_0`.
    fn encode0<E: ExtensionField<F>>(&self, m: &[E]) -> Vec<E>;

    /// Encodes `m * G_0` into `w`.
    fn batch_encode0_into(&self, m: RowMajorMatrixView<F>, w: RowMajorMatrixViewMut<F>);

    /// Returns `w = m * G_d` where `m = interpolate(polys)`.
    fn batch_encode(&self, polys: RowMajorMatrixView<F>) -> RowMajorMatrix<F>;

    /// Returns `interpolate((diag(T_i)[j], w[j]), (diag(T_i')[j], w[j+n_i])))`
    fn fold<E: ExtensionField<F>>(&self, i: usize, w: Vec<E>, r_i: E) -> Vec<E>;

    /// Returns `interpolate((diag(T_i)[j], l), (diag(T_i')[j], r)))`
    fn interpolate<E: ExtensionField<F>>(&self, i: usize, j: usize, l: E, r: E, r_i: E) -> E;
}

#[derive(Clone, Debug)]
pub struct GenericRandomFoldableCode<F> {
    log2_c: usize,
    d: usize,
    log2_k_0: usize,
    g_0: Vec<Vec<F>>,
    ts: Vec<Vec<F>>,
    weights: Vec<Vec<F>>,
}

impl<F: Field + FromUniformBytes> GenericRandomFoldableCode<F> {
    pub fn reed_solomon_g_0_with_random_ts(
        log2_c: usize,
        log2_k_0: usize,
        d: usize,
        mut rng: impl RngCore,
    ) -> Self {
        let k_0 = 1 << log2_k_0;
        let n_0 = k_0 << log2_c;
        let g_0 = (0..n_0)
            .map(|i| F::from_canonical_usize(i).powers().take(k_0).collect())
            .collect();
        let ts = (0..d)
            .map(|i| F::random_vec(n_0 << i, &mut rng))
            .collect_vec();
        let weights = ts
            .iter()
            .map(|t_i| {
                batch_multiplicative_inverse(&t_i.iter().map(|t_i_j| -t_i_j.double()).collect_vec())
            })
            .collect();
        Self {
            log2_c,
            d,
            log2_k_0,
            g_0,
            ts,
            weights,
        }
    }
}

impl<F: Field> RandomFoldableCode<F> for GenericRandomFoldableCode<F> {
    fn log2_c(&self) -> usize {
        self.log2_c
    }

    fn log2_k_0(&self) -> usize {
        self.log2_k_0
    }

    fn d(&self) -> usize {
        self.d
    }

    fn ts(&self) -> &[Vec<F>] {
        &self.ts
    }

    fn encode0<E: ExtensionField<F>>(&self, m: &[E]) -> Vec<E> {
        self.g_0
            .iter()
            .map(|g_0_i| izip!(m, g_0_i).map(|(m_j, g_0_i_j)| *m_j * *g_0_i_j).sum())
            .collect()
    }

    fn batch_encode0_into(&self, m: RowMajorMatrixView<F>, mut w: RowMajorMatrixViewMut<F>) {
        debug_assert_eq!(m.height(), self.k_0());
        debug_assert_eq!(w.height(), self.n_0());

        // i-th reed-solomon evaluation
        izip!(&self.g_0, w.rows_mut()).for_each(|(g_0_i, mut w_i)| {
            // j-th message coefficient
            izip!(m.row_slices(), g_0_i)
                .for_each(|(m_j, g_0_i_j)| w_i.slice_add_scaled_assign(m_j, *g_0_i_j))
        });
    }

    fn batch_encode(&self, polys: RowMajorMatrixView<F>) -> RowMajorMatrix<F> {
        debug_assert_eq!(polys.height(), self.k_d());

        let m = batch_interpolate(polys.as_cow());
        let mut w = RowMajorMatrix::new(F::zero_vec(m.width() * self.n_d()), m.width());

        // d = 0
        izip_par!(
            m.par_row_chunks(self.k_0()),
            w.par_row_chunks_mut(self.n_0())
        )
        .for_each(|(m, w)| self.batch_encode0_into(m, w));

        // d in 1..d
        (1..=self.d()).for_each(|i| {
            let n_i = self.n_0() << i;
            w.par_row_chunks_mut(n_i).for_each(|mut w| {
                let (mut l, mut r) = w.split_rows_mut(n_i >> 1);
                izip!(l.rows_mut(), r.rows_mut(), &self.ts()[i - 1])
                    .for_each(|(mut l, r, t)| l.slice_dit_butterfly(r, t))
            })
        });

        w
    }

    fn fold<E: ExtensionField<F>>(&self, i: usize, mut w: Vec<E>, r_i: E) -> Vec<E> {
        debug_assert_eq!((w.len() / self.n_0()).ilog2() as usize - 1, i);

        let mid = w.len() / 2;
        let (l, r) = w.split_at_mut(mid);
        izip!(l, r, &self.ts()[i], &self.weights[i])
            .for_each(|(l, r, t, weight)| *l += (r_i - *t) * (*r - *l) * *weight);
        w.truncate(mid);
        w
    }

    fn interpolate<E: ExtensionField<F>>(&self, i: usize, j: usize, l: E, r: E, r_i: E) -> E {
        let t = self.ts()[i][j];
        let weight = self.weights[i][j];
        l + (r_i - t) * (r - l) * weight
    }
}
