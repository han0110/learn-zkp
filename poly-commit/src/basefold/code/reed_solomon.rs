use crate::basefold::code::RandomFoldableCode;
use core::fmt::Debug;
use p3::{
    dft::{Radix2DitParallel, TwoAdicSubgroupDft},
    field::{batch_multiplicative_inverse, ExtensionField, TwoAdicField},
    matrix::{bitrev::BitReversableMatrix, dense::RowMajorMatrix, Matrix},
};
use util::Itertools;

#[derive(Clone, Debug)]
pub struct ReedSolomonCode<F> {
    lambda: usize,
    log2_c: usize,
    d: usize,
    t_inv_halves: Vec<Vec<F>>,
    fft: Radix2DitParallel<F>,
}

impl<F: TwoAdicField> ReedSolomonCode<F> {
    pub fn new(lambda: usize, log2_c: usize, d: usize) -> Self {
        let n_0 = 1 << log2_c;
        let t_inv_halves = (0..d)
            .map(|i| {
                let t_i = F::two_adic_generator(log2_c + i + 1)
                    .powers()
                    .take(n_0 << i)
                    .collect_vec();
                batch_multiplicative_inverse(&t_i.iter().map(F::double).collect_vec())
            })
            .collect();
        Self {
            lambda,
            log2_c,
            d,
            t_inv_halves,
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

    fn num_queries(&self) -> usize {
        // Delta in unique decoding regime.
        // TODO: Adapt delta in list-decoding regime.
        let relative_minimum_distance = 1.0 - (1.0 / self.c() as f64);
        let delta = 1.0 - relative_minimum_distance / 3.0;
        (self.lambda() as f64 / -delta.log2()).ceil() as usize
    }

    fn encode0<E: ExtensionField<F>>(&self, m: &[E]) -> Vec<E> {
        debug_assert_eq!(m.len(), 1);

        vec![m[0]; self.c()]
    }

    fn encode_batch(&self, m: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let mut m = m.bit_reverse_rows().to_row_major_matrix();
        m.pad_to_height(self.n_d(), F::ZERO);
        self.fft.dft_batch(m).to_row_major_matrix()
    }

    fn t_inv_halves(&self, i: usize) -> &[F] {
        &self.t_inv_halves[i]
    }
}
