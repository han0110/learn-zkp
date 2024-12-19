use crate::basefold::code::RandomFoldableCode;
use core::fmt::Debug;
use p3::{
    dft::{Radix2DitParallel, TwoAdicSubgroupDft},
    field::{batch_multiplicative_inverse, ExtensionField, FromUniformBytes, TwoAdicField},
    matrix::{bitrev::BitReversableMatrix, dense::RowMajorMatrix, Matrix},
};
use util::{izip, Itertools};

#[derive(Clone, Debug)]
pub struct ReedSolomonCode<F> {
    lambda: usize,
    log2_c: usize,
    d: usize,
    ts: Vec<Vec<F>>,
    weights: Vec<Vec<F>>,
    fft: Radix2DitParallel<F>,
}

impl<F: TwoAdicField + FromUniformBytes> ReedSolomonCode<F> {
    pub fn new(lambda: usize, log2_c: usize, d: usize) -> Self {
        let n_0 = 1 << log2_c;
        let ts = (0..d)
            .map(|i| {
                F::two_adic_generator(log2_c + i + 1)
                    .powers()
                    .take(n_0 << i)
                    .collect_vec()
            })
            .collect_vec();
        let weights = ts
            .iter()
            .map(|t_i| {
                batch_multiplicative_inverse(&t_i.iter().map(|t_i_j| -t_i_j.double()).collect_vec())
            })
            .collect();
        Self {
            lambda,
            log2_c,
            d,
            ts,
            weights,
            fft: Default::default(),
        }
    }
}

impl<F: TwoAdicField + Ord> RandomFoldableCode<F> for ReedSolomonCode<F> {
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

    fn relative_minimum_distance(&self) -> f64 {
        1.0 - (1.0 / self.c() as f64)
    }

    fn num_queries(&self) -> usize {
        let delta = 1.0 - self.relative_minimum_distance() / 3.0;
        (self.lambda() as f64 / -delta.log2()).ceil() as usize
    }

    fn ts(&self) -> &[Vec<F>] {
        &self.ts
    }

    fn encode0<E: ExtensionField<F>>(&self, m: &[E]) -> Vec<E> {
        debug_assert_eq!(m.len(), 1);

        vec![m[0]; self.c()]
    }

    fn batch_encode(&self, m: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let mut m = m.bit_reverse_rows().to_row_major_matrix();
        m.pad_to_height(self.n_d(), F::ZERO);
        self.fft.dft_batch(m).to_row_major_matrix()
    }

    fn fold<E: ExtensionField<F>>(&self, i: usize, w: &mut Vec<E>, r_i: E) {
        debug_assert_eq!((w.len() / self.n_0()).ilog2() as usize - 1, i);

        let mid = w.len() / 2;
        let (l, r) = w.split_at_mut(mid);
        izip!(l, r, &self.ts()[i], &self.weights[i])
            .for_each(|(l, r, t, weight)| *l += (r_i - *t) * (*r - *l) * *weight);
        w.truncate(mid);
    }

    fn interpolate<E: ExtensionField<F>>(&self, i: usize, j: usize, l: E, r: E, r_i: E) -> E {
        let t = self.ts()[i][j];
        let weight = self.weights[i][j];
        l + (r_i - t) * (r - l) * weight
    }
}
